"""Implements actions used by doit tasks"""

import inspect
import io
import os
import pdb
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import partial
from io import BytesIO, StringIO
from multiprocessing import Process
from pathlib import PurePath
from threading import Thread
from typing import IO, TYPE_CHECKING, Any, Literal, LiteralString, TextIO, override

from ._type_guards import is_object_list, is_str_list
from .exceptions import InvalidTask, TaskError, TaskFailed

if TYPE_CHECKING:
    from .task import Task
type NormalizedCallable[T] = tuple[Callable[..., T], list[Any], dict[str, Any]]
type RawCallable[T] = (
    Callable[[], T]
    | tuple[Callable[[], T]]
    | tuple[Callable[..., T], list[Any]]
    | NormalizedCallable[T]
)


def normalize_callable[T](ref: RawCallable[T]) -> NormalizedCallable[T]:
    """return a list with (callable, *args, **kwargs)
    ref can be a simple callable or a tuple
    """
    match ref:
        case (_, _, _):
            return ref
        case (callable, args):
            return (callable, args, {})
        case (callable,) | callable:
            return (callable, [], {})


# Actions
class BaseAction(ABC):
    """Base class for all actions"""

    task: "Task | None" = None

    @abstractmethod
    def execute(
        self, out: IO[Any] | None = None, err: IO[Any] | None = None
    ) -> TaskError | TaskFailed | None: ...

    @staticmethod
    def _prepare_kwargs(
        task: "Task | None",
        func: Callable[..., Any],
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ):
        """
        Prepare keyword arguments (targets, dependencies, changed,
        cmd line options)
        Inspect python callable and add missing arguments:
        - that the callable expects
        - have not been passed (as a regular arg or as keyword arg)
        - are available internally through the task object
        """
        # Return just what was passed in task generator
        # dictionary if the task isn't available
        if not task:
            return kwargs

        func_sig = inspect.signature(func)
        sig_params = func_sig.parameters.values()
        func_has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig_params)

        # use task meta information as extra_args
        meta_args = {
            "task": lambda: task,
            "targets": lambda: list(task.targets),
            "dependencies": lambda: list(task.file_dep),
            "changed": lambda: list(task.dep_changed),
        }

        # start with dict passed together on action definition
        kwargs = kwargs.copy()
        bound_args = func_sig.bind_partial(*args)

        # add meta_args
        for key in meta_args.keys():
            # check key is a positional parameter
            if key in func_sig.parameters:
                sig_param = func_sig.parameters[key]

                # it is forbidden to use default values for this arguments
                # because the user might be unaware of this magic.
                if sig_param.default != sig_param.empty:
                    msg = (
                        f"Task {task.name}, action {func.__name__}():"
                        f"The argument '{key}' is not allowed to have "
                        "a default value (reserved by doit)"
                    )
                    raise InvalidTask(msg)

                # if value not taken from position parameter
                if key not in bound_args.arguments:
                    kwargs[key] = meta_args[key]()

        # add tasks parameter options
        opt_args: dict[str, Any] = dict(task.options or {})
        if task.pos_arg is not None:
            opt_args[task.pos_arg] = task.pos_arg_val

        for key in opt_args.keys():
            # check key is a positional parameter
            if key in func_sig.parameters:
                # if value not taken from position parameter
                if key not in bound_args.arguments:
                    kwargs[key] = opt_args[key]

            # if function has **kwargs include extra_arg on it
            elif func_has_kwargs and key not in kwargs:
                kwargs[key] = opt_args[key]
        return kwargs


class CmdAction(BaseAction):
    """
    Command line action. Spawns a new process.

    @ivar action(str,list,callable): subprocess command string or string list,
         see subprocess.Popen first argument.
         It may also be a callable that generates the command string.
         Strings may contain python mappings with the keys: dependencies,
         changed and targets. ie. "zip %(targets)s %(changed)s"
    @ivar task(Task): reference to task that contains this action
    @ivar save_out: (str) name used to save output in `values`
    @ivar shell: use shell to execute command
                 see subprocess.Popen `shell` attribute
    @ivar encoding (str): encoding of the process output
    @ivar decode_error (str): value for decode() `errors` param
                              while decoding process output
    @ivar pkwargs: Popen arguments except 'stdout' and 'stderr'
    """

    STRING_FORMAT: Literal["old", "new", "both"] = "old"

    type _Action = str | list[str] | RawCallable[object]

    _action: _Action
    task: "Task | None"
    out: str | None
    err: str | None
    result: str | None
    values: dict[str | None, Any]
    save_out: str | None
    shell: bool
    encoding: LiteralString
    decode_error: LiteralString
    buffering: int

    def __init__(
        self,
        action: str | list[str] | RawCallable[object],
        task: "Task | None" = None,
        save_out: str | None = None,
        shell: bool = True,
        encoding: LiteralString = "utf-8",
        decode_error: LiteralString = "replace",
        buffering: int = 0,
        **pkwargs: Any,  # pyright: ignore[reportAny]
    ):  # pylint: disable=W0231
        """
        :ivar buffering: (int) stdout/stderr buffering.
               Not to be confused with subprocess buffering
               -   0 -> line buffering
               -   positive int -> number of bytes
        """
        for forbidden in ("stdout", "stderr"):
            if forbidden in pkwargs:
                msg = "CmdAction can't take param named '{0}'."
                raise InvalidTask(msg.format(forbidden))
        self._action = action
        self.task = task
        self.out = None
        self.err = None
        self.result = None
        self.values = {}
        self.save_out = save_out
        self.shell = shell
        self.encoding = encoding
        self.decode_error = decode_error
        self.pkwargs: dict[str, Any] = pkwargs
        self.buffering = buffering

    @property
    def action(self) -> str | list[str]:
        if isinstance(self._action, (str, list)):
            return self._action
        else:
            # action can be a callable that returns a string command
            ref, args, kw = normalize_callable(self._action)
            kwargs = self._prepare_kwargs(self.task, ref, args, kw)
            res = ref(*args, **kwargs)
            assert isinstance(res, str)
            return res

    @action.setter
    def action(self, value: _Action) -> None:
        self._action = value

    def _print_process_output(
        self, process: Process, input_: BytesIO, capture: TextIO, realtime: TextIO
    ):
        """Reads 'input_' until process is terminated.
        Writes 'input_' content to 'capture' (string)
        and 'realtime' stream
        """
        if self.buffering:
            read = partial(input_.read, self.buffering)
        else:
            # line buffered
            read = input_.readline
        while True:
            try:
                line = read().decode(self.encoding, self.decode_error)
            except Exception:
                # happens when fails to decoded input
                process.terminate()
                _ = input_.read()
                raise
            if not line:
                break
            _ = capture.write(line)
            if realtime:
                _ = realtime.write(line)
                realtime.flush()  # required if on byte buffering mode

    @override
    def execute(self, out: IO[Any] | None = None, err: IO[Any] | None = None):
        """
        Execute command action

        both stdout and stderr from the command are captured and saved
        on self.out/err. Real time output is controlled by parameters
        @param out: None - no real time output
                    a file like object (has write method)
        @param err: idem
        @return failure:
            - None: if successful
            - TaskError: If subprocess return code is greater than 125
            - TaskFailed: If subprocess return code isn't zero (and
              not greater than 125)
        """
        try:
            action = self.expand_action()
        except Exception as exc:
            return TaskError("CmdAction Error creating command string", exc)

        # set environ to change output buffering
        subprocess_pkwargs = self.pkwargs.copy()
        env: dict[str, str] | None = None
        if "env" in subprocess_pkwargs:
            env = subprocess_pkwargs["env"]
            assert isinstance(env, dict)
            del subprocess_pkwargs["env"]
        if self.buffering:
            if not env:
                env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

        capture_io = self.task.io.capture if self.task else True
        if capture_io:
            p_out = p_err = subprocess.PIPE
        else:
            if capture_io is False:
                p_out = out
                p_err = err
            else:  # None
                p_out = p_err = open(os.devnull, "w")

        # spawn task process
        process = subprocess.Popen(
            action,
            shell=self.shell,
            # bufsize=2, # ??? no effect use PYTHONUNBUFFERED instead
            stdout=p_out,
            stderr=p_err,
            env=env,
            **subprocess_pkwargs,  # pyright: ignore[reportAny]
        )

        if capture_io:
            output = StringIO()
            errput = StringIO()
            t_out = Thread(
                target=self._print_process_output,
                args=(process, process.stdout, output, out),
            )
            t_err = Thread(
                target=self._print_process_output,
                args=(process, process.stderr, errput, err),
            )
            t_out.start()
            t_err.start()
            t_out.join()
            t_err.join()

            self.out = output.getvalue()
            self.err = errput.getvalue()
            self.result = self.out + self.err

        # make sure process really terminated
        return_code = process.wait()

        # task error - based on:
        # http://www.gnu.org/software/bash/manual/bashref.html#Exit-Status
        # it doesnt make so much difference to return as Error or Failed anyway
        if return_code > 125:
            return TaskError(
                "Command error: '%s' returned %s" % (action, process.returncode)
            )

        # task failure
        if return_code != 0:
            return TaskFailed(
                "Command failed: '%s' returned %s" % (action, process.returncode)
            )

        # save stdout in values
        if self.save_out:
            self.values[self.save_out] = self.out

    def expand_action(self) -> str | list[str]:
        """Expand action using task meta informations if action is a string.
        Convert `Path` elements to `str` if action is a list.
        @returns: string -> expanded string if action is a string
                  list - string -> expanded list of command elements
        """
        if not self.task:
            return self.action

        # cant expand keywords if action is a list of strings
        if is_object_list(self.action):
            action: list[str] = []
            for element in self.action:
                if isinstance(element, str):
                    action.append(element)
                elif isinstance(element, PurePath):
                    action.append(str(element))
                else:
                    msg = (
                        "%s. CmdAction element must be a str "
                        "or Path from pathlib. Got '%r' (%s)"
                    )
                    raise InvalidTask(msg % (self.task.name, element, type(element)))
            return action

        subs_dict = {
            "targets": " ".join(self.task.targets),
            "dependencies": " ".join(self.task.file_dep),
        }

        # dep_changed is set on get_status()
        # Some commands (like `clean` also uses expand_args but do not
        # uses get_status, so `changed` is not available.
        if self.task.dep_changed:
            subs_dict["changed"] = " ".join(self.task.dep_changed)

        # task option parameters
        subs_dict.update(self.task.options or {})
        # convert positional parameters from list space-separated string
        if self.task.pos_arg:
            if self.task.pos_arg_val:
                pos_val = " ".join(self.task.pos_arg_val)
            else:
                pos_val = ""
            subs_dict[self.task.pos_arg] = pos_val

        assert isinstance(self.action, str)

        if self.STRING_FORMAT == "old":
            return self.action % subs_dict
        elif self.STRING_FORMAT == "new":
            return self.action.format(**subs_dict)
        else:
            assert self.STRING_FORMAT == "both"
            return self.action.format(**subs_dict) % subs_dict

    @override
    def __str__(self):
        return "Cmd: %s" % self._action

    @override
    def __repr__(self):
        return "<CmdAction: '%s'>" % str(self._action)


class Writer(object):
    """Write to N streams.

    This is used on python-actions to allow the stream to be output to terminal
    and captured at the same time.
    """

    def __init__(self, *writers: IO[Any]):
        """@param writers - file stream like objects"""
        self.writers: list[IO[Any]] = []
        self.orig_stream: IO[Any] | None = None  # The original stream terminal/file
        for writer in writers:
            self.add_writer(writer)

    def add_writer(self, stream: IO[Any], *, is_original: bool = False):
        """adds a stream to the list of writers
        @param is_original: (bool) if specified overwrites real isatty from stream
        """
        self.writers.append(stream)
        if is_original:
            self.orig_stream = stream

    def write(self, text: str):
        """write 'text' to all streams"""
        for stream in self.writers:
            _ = stream.write(text)

    def flush(self):
        """flush all streams"""
        for stream in self.writers:
            stream.flush()

    def isatty(self):
        if self.orig_stream:
            return self.orig_stream.isatty()
        return False

    def fileno(self):
        if self.orig_stream:
            return self.orig_stream.fileno()
        raise io.UnsupportedOperation()


class PythonAction(BaseAction):
    """Python action. Execute a python callable.

    @ivar py_callable: (callable) Python callable
    @ivar args: (sequence)  Extra arguments to be passed to py_callable
    @ivar kwargs: (dict) Extra keyword arguments to be passed to py_callable
    @ivar task(Task): reference to task that contains this action
    @ivar pm_pdb: if True drop into PDB on exception when executing task
    """

    pm_pdb: bool = False

    def __init__(
        self,
        py_callable: Callable[..., Any],
        args: list[object] | tuple[object, ...] | None = None,
        kwargs: dict[str, object] | None = None,
        task: "Task | None" = None,
    ):
        # pylint: disable=W0231
        self.py_callable: Callable[[], object] = py_callable
        self.task: "Task | None" = task
        self.out: str | None = None
        self.err: str | None = None
        self.result: dict[str, object] | str | None = None
        self.values: dict[str, object] = {}

        self.args: list[object] | tuple[object, ...]
        self.kwargs: dict[str, object]

        if args is None:
            self.args = []
        else:
            self.args = args

        if kwargs is None:
            self.kwargs = {}
        else:
            self.kwargs = kwargs

        # check valid parameters
        if not hasattr(self.py_callable, "__call__"):
            msg = "%r PythonAction must be a 'callable' got %r."
            raise InvalidTask(msg % (self.task, self.py_callable))
        if inspect.isclass(self.py_callable):
            msg = "%r PythonAction can not be a class got %r."
            raise InvalidTask(msg % (self.task, self.py_callable))
        if inspect.isbuiltin(self.py_callable):
            msg = "%r PythonAction can not be a built-in got %r."
            raise InvalidTask(msg % (self.task, self.py_callable))
        if type(self.args) is not tuple and type(self.args) is not list:
            msg = "%r args must be a 'tuple' or a 'list'. got '%s'."
            raise InvalidTask(msg % (self.task, self.args))
        if type(self.kwargs) is not dict:
            msg = "%r kwargs must be a 'dict'. got '%s'"
            raise InvalidTask(msg % (self.task, self.kwargs))

    def _prepare_kwargs(self):
        return BaseAction._prepare_kwargs(
            self.task, self.py_callable, self.args, self.kwargs
        )

    @contextmanager
    def __wrap_captured_io(self, out: IO[Any] | None = None, err: IO[Any] | None = None):
        # set std stream
        old_stdout = sys.stdout
        output = StringIO()
        old_stderr = sys.stderr
        errput = StringIO()

        try:
            out_writer = Writer()
            # capture output but preserve isatty() from original stream
            out_writer.add_writer(output)
            if out:
                out_writer.add_writer(out, is_original=True)
            sys.stdout = out_writer

            err_writer = Writer()
            err_writer.add_writer(errput)
            if err:
                err_writer.add_writer(err, is_original=True)
            sys.stderr = err_writer

            yield

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.out = output.getvalue()
            self.err = errput.getvalue()

    @contextmanager
    def __wrap_uncaptured_io(
        self, out: IO[Any] | None = None, err: IO[Any] | None = None
    ):
        old_stdout: TextIO | None = None
        old_stderr: TextIO | None = None
        if out:
            old_stdout = sys.stdout
            sys.stdout = out
        if err:
            old_stderr = sys.stderr
            sys.stderr = err

        try:
            yield
        finally:
            if out:
                sys.stdout = old_stdout
            if err:
                sys.stderr = old_stderr

    def __wrap_execution_io(self, out: IO[Any] | None = None, err: IO[Any] | None = None):
        capture_io = self.task.io.capture if self.task else True
        if capture_io:
            return self.__wrap_captured_io(out, err)
        else:
            return self.__wrap_uncaptured_io(out, err)

    @override
    def execute(self, out: IO[Any] | None = None, err: IO[Any] | None = None):
        """Execute command action

        both stdout and stderr from the command are captured and saved
        on self.out/err. Real time output is controlled by parameters
        @param out: None - no real time output
                    a file like object (has write method)
        @param err: idem

        @return failure: see CmdAction.execute
        """

        with self.__wrap_execution_io(out, err):
            kwargs = self._prepare_kwargs()

            # execute action / callable
            try:
                returned_value = self.py_callable(*self.args, **kwargs)
            except Exception as exception:
                if self.pm_pdb:  # pragma: no cover
                    # start post-mortem debugger
                    deb = pdb.Pdb(stdin=sys.__stdin__, stdout=sys.__stdout__)
                    deb.reset()
                    deb.interaction(None, sys.exc_info()[2])
                return TaskError("PythonAction Error", exception)

        # if callable returns false. Task failed
        if returned_value is False:
            return TaskFailed(
                "Python Task failed: '%s' returned %s"
                % (self.py_callable, returned_value)
            )
        elif returned_value is True or returned_value is None:
            pass
        elif isinstance(returned_value, str):
            self.result = returned_value
        elif isinstance(returned_value, dict):
            self.values = returned_value
            self.result = returned_value
        elif isinstance(returned_value, (TaskFailed, TaskError)):
            return returned_value
        else:
            return TaskError(
                "Python Task error: '%s'. It must return:\n"
                "False for failed task.\n"
                "True, None, string or dict for successful task\n"
                "returned %s (%s)"
                % (self.py_callable, returned_value, type(returned_value))
            )

    @override
    def __str__(self):
        # get object description excluding runtime memory address
        return "Python: %s" % str(self.py_callable)[1:].split(" at ")[0]

    @override
    def __repr__(self):
        return "<PythonAction: '%s'>" % (repr(self.py_callable))


def create_action(action: object, task_ref: "Task", param_name: str) -> BaseAction:
    """
    Create action using proper constructor based on the parameter type

    @param action: Action to be created
    @type action: L{BaseAction} subclass object, str, tuple or callable
    @param task_ref: Task object this action belongs to
    @param param_name: str, name of task param. i.e actions, teardown, clean
    @raise InvalidTask: If action parameter type isn't valid
    """

    args: list[object]
    kwargs: dict[str, object]

    match action:
        case BaseAction():
            action.task = task_ref
            return action
        case str():
            return CmdAction(action, task_ref, shell=True)
        case action if is_str_list(action):
            return CmdAction(action, task_ref, shell=False)
        case tuple() if len(action) > 3:
            msg = "Task '{}': invalid '{}' tuple length. got: {!r} {}".format(
                task_ref.name, param_name, action, type(action)
            )
            raise InvalidTask(msg)
        case tuple():
            py_callable, args, kwargs = list(action) + [None] * (3 - len(action))
            return PythonAction(py_callable, args, kwargs, task_ref)
        case Callable() as action:
            return PythonAction(action, task=task_ref)
        case _:
            msg = "Task '{}': invalid '{}' type. got: {!r} {}".format(
                task_ref.name, param_name, action, type(action)
            )
            raise InvalidTask(msg)
