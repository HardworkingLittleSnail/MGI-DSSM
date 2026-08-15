import os
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    os.environ["REMOTE_HOST"],
    port=int(os.environ["REMOTE_PORT"]),
    username=os.environ["REMOTE_USER"],
    password=os.environ["REMOTE_PASSWORD"],
    timeout=20,
)
command = r'''cd /root/autodl-tmp/version2026draft && find . \( -path './outputs' -o -path './data' -o -path './.git' -o -path './__pycache__' \) -prune -o -type f \( -iname '*autoformer*' -o -iname '*itransformer*' -o -path './Compare-Models/layers/*' -o -path './Compare-Models/models/*' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort'''
_, stdout, stderr = client.exec_command(command, timeout=60)
print(stdout.read().decode("utf-8", errors="replace"))
error = stderr.read().decode("utf-8", errors="replace")
if error:
    print(error)
client.close()
