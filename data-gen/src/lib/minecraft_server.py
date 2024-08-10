from typing import Dict, Iterable
from contextlib import contextmanager
import time
import re
import subprocess
import logging

from mcrcon import MCRcon, MCRconException

from .util import timeAndLog, timeLimitUnsafe, withRetries


logger = logging.getLogger(__name__)


def parseServerProperties(text: str):
    """Parses a Minecraft server.properties file"""
    result: Dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


class MinecraftServer:
    # ==============================================================================================

    def __init__(
        self,
        dirPath: str,
        jarPath: str,
        javaArgs:   Iterable[str] = ("-Xmx2G", "-Xms2G"),
        serverArgs: Iterable[str] = ()
    ):
        """if <jarPath> is relative, it is relative to <dirPath>"""
        self._dirPath = dirPath
        self._jarPath = jarPath
        self.javaArgs = javaArgs
        self.serverArgs = serverArgs

        with open(f"{dirPath}/server.properties", "r", encoding="utf-8") as file:
            self._serverProperties = parseServerProperties(file.read())
        self._isRconEnabled = self._serverProperties["enable-rcon"] == "true"

        self._process = None
        self._rconClient = None

    def __del__(self):
        self.stop()

    # ==============================================================================================

    @property
    def dirPath(self):
        return self._dirPath

    @property
    def jarPath(self):
        return self._jarPath

    @property
    def javaArgs(self):
        return self._javaArgs

    @javaArgs.setter
    def javaArgs(self, value: Iterable[str]):
        self._javaArgs = list(value)

    @property
    def serverArgs(self):
        return self._serverArgs

    @serverArgs.setter
    def serverArgs(self, value: Iterable[str]):
        self._serverArgs = list(value)

    @property
    def serverProperties(self):
        return self._serverProperties

    @property
    def isRconEnabled(self):
        return self._isRconEnabled

    @property
    def worldPath(self):
        return f"{self._dirPath}/{self._serverProperties['level-name']}"

    # ==============================================================================================

    @property
    def isRunning(self):
        return self._process is not None and self._process.poll() is None

    def start(self):
        if self.isRunning:
            return
        with timeAndLog(logger, "Starting server...", "Server started in %.2fs"):
            self._start()

    def _start(self):
        self._process = subprocess.Popen(
            ["java"] + self.javaArgs + ["-jar", self._jarPath, "-nogui"] + self.serverArgs,
            cwd=self._dirPath,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )

        if self._isRconEnabled:
            targetLineRegex = re.compile(r"\[.*\] \[minecraft/MainThread\]: RCON running on 0\.0\.0\.0:" + self._serverProperties["rcon.port"])
        else:
            targetLineRegex = re.compile(r"\[.*\] \[minecraft/DedicatedServer\]: Done \(.*\)! For help, type \"help\"")

        while True:
            try:
                with timeLimitUnsafe(60 * 5):
                    line = self._process.stdout.readline()
            except TimeoutError as e:
                raise RuntimeError("Server failed to start in time!") from e
            if line == "":
                raise RuntimeError("Server failed to start!")
            logger.debug("Server: %s", line.rstrip())
            if re.match(targetLineRegex, line) is not None:
                break
        time.sleep(1) # We need to wait a bit longer for the server to be completely ready

        if self._isRconEnabled:
            # It is possible for the server to crash, stopping the process without going through
            # stop(). In that case, the RCON client might still be connected, so we make sure to
            # disconnect it here.
            if self._rconClient is not None:
                self._rconClient.disconnect()
            self._rconClient = MCRcon("localhost", self.serverProperties["rcon.password"], int(self._serverProperties["rcon.port"]), timeout=60)
            self._rconClient.connect()

    def stop(self, timeoutSeconds: float = 7, terminateTimeoutSeconds: float = 30):
        if not self.isRunning:
            return
        with timeAndLog(logger, "Stopping server...", "Server stopped in %.2fs"):
            self._stop(timeoutSeconds, terminateTimeoutSeconds)

    def _stop(self, timeoutSeconds: float, terminateTimeoutSeconds: float):
        if self._rconClient is not None:
            self._rconClient.disconnect()
        self.runCommandStdin("stop")
        try:
            self._process.wait(timeoutSeconds)
        except subprocess.TimeoutExpired:
            logger.warning("Server failed to stop within %.2fs. Sending SIGTERM.", timeoutSeconds)
            self._process.terminate()
            try:
                self._process.wait(terminateTimeoutSeconds)
            except subprocess.TimeoutExpired:
                logger.error("Server failed to terminate within %.2fs. Sending SIGKILL.", terminateTimeoutSeconds)
                self._process.kill()
                self._process.wait()

    @contextmanager
    def running(self):
        if self.isRunning:
            yield
            return
        self.start()
        try:
            yield
        finally:
            self.stop()

    # ==============================================================================================

    def runCommandRcon(self, command: str):
        if not self.isRunning:
            raise RuntimeError("Server is not running!")
        if not self.isRconEnabled:
            raise RuntimeError("RCON is not enabled for this server!")

        def onRetry(e, retriesLeft):
            logger.warning("RCON command failed (%i retries left). Restarting the server...", retriesLeft, exc_info=True)
            self.stop()
            self.start()

        return withRetries(
            lambda: self._rconClient.command(command),
            MCRconException, 2, onRetry
        )

    def runCommandStdin(self, command: str):
        if not self.isRunning:
            raise RuntimeError("Server is not running!")
        self._process.stdin.write(f"{command}\n")
        self._process.stdin.flush()

    @property
    def stdout(self):
        if not self.isRunning:
            raise RuntimeError("Server is not running!")
        return self._process.stdout
