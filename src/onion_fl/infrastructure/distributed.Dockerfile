# A machine for the distributed backend (I14): the emulated backend's host
# image (Python and the runtime dependencies of onion_fl, pinned to the
# launcher's numeric versions) with an SSH server. It is how a Linux server, or
# the development PC through Docker, joins a distributed run, and what the
# integration test starts several of to act as machines. Build it on the
# machine's own architecture (a Raspberry Pi builds an ARM64 one):
#
#   docker build -t onionfl-distributed --build-arg BASE=onionfl-emulated:latest - \
#     < src/onion_fl/infrastructure/distributed.Dockerfile
#
# The onion_fl source is mounted (or copied) at /onionfl/src; an SSH session
# does not inherit the image's environment, so the inventory sets PYTHONPATH.
# The public key that may log in comes from AUTHORIZED_KEY when the container
# starts: no key is ever baked into the image.
ARG BASE=onionfl-emulated:latest
FROM ${BASE}
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd
EXPOSE 22
CMD ["sh", "-c", "mkdir -p /root/.ssh && printf '%s\\n' \"$AUTHORIZED_KEY\" > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && ssh-keygen -A >/dev/null && exec /usr/sbin/sshd -D -e"]
