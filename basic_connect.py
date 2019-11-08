from subprocess import Popen, PIPE
import os

base_dir_path = os.path.dirname(os.path.realpath(__file__))
commands = [
    "java",
    "-jar",
    "/home/toby/microrts_env/microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar",
    "more",
    "options"

]
process = Popen(
    commands,
    stdin=PIPE,
    stdout=PIPE
)
stdout, stderr = process.communicate()
print(stdout)
print(stderr)
