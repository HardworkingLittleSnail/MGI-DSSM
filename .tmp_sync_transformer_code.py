import hashlib
import os
import posixpath
from pathlib import Path

import paramiko

project = "/root/autodl-tmp/version2026draft"
local_project = Path(__file__).resolve().parent
fixed_files = [
    "comparison_protocol.py",
    "Compare-Models/run_autoformer_itransformer.py",
    "train_autoformer_itransformer_version3_10seeds.sh",
    "Compare-Models/Autoformer/models/Autoformer.py",
    "Compare-Models/iTransformer/model/iTransformer.py",
]
layer_dirs = [
    "Compare-Models/Autoformer/layers",
    "Compare-Models/iTransformer/layers",
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    os.environ["REMOTE_HOST"],
    port=int(os.environ["REMOTE_PORT"]),
    username=os.environ["REMOTE_USER"],
    password=os.environ["REMOTE_PASSWORD"],
    timeout=20,
)
sftp = client.open_sftp()
files = list(fixed_files)
for directory in layer_dirs:
    for name in sftp.listdir(posixpath.join(project, directory)):
        if name.endswith(".py"):
            files.append(posixpath.join(directory, name))

for relative in sorted(set(files)):
    remote = posixpath.join(project, relative.replace("\\", "/"))
    local = local_project / Path(relative)
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    remote_hash = hashlib.sha256(sftp.open(remote, "rb").read()).hexdigest()
    local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
    status = "OK" if remote_hash == local_hash else "MISMATCH"
    print(f"{status}\t{relative}\t{local_hash}")

sftp.close()
client.close()
