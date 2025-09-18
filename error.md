=> => extracting sha256:367a73a92c9a9463eff8ca21862e88019db16  13.6s
 => => extracting sha256:d56f0a5f3819d50b854c8111201d1f191e1ae6  0.0s
 => => extracting sha256:9793c1110e27d2043f01be0fe59a0b1985af08  0.0s
 => => extracting sha256:e6247aec3b52e76c3dee54eacec6e02f50e54b  0.0s
 => => extracting sha256:cf022c09446bcc35ba3211a76dd674294fbd14  0.0s
 => => extracting sha256:1bcca1587e3fd3540400bf220693f188a4b792  0.0s
 => => extracting sha256:1d15e9b4818551733db19c8ef2105bd2488cc4  0.0s
 => => extracting sha256:cd12bd996414fe1b18e68f32de5a00d11f8b9d  0.0s
 => => extracting sha256:34dc5cb1edd78b8cfcdfe1aa24d29a2d44349a  0.0s
 => => extracting sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cd  0.0s
 => => extracting sha256:710bef11f5bb4f6249ccfc9f18fd23ccf33e26  0.0s
 => ERROR [2/7] RUN apt-get update && apt-get install -y     gc  2.5s
------
 > [2/7] RUN apt-get update && apt-get install -y     gcc     g++     make     pkg-config     libfreetype6-dev     libpng-dev     libfontconfig1-dev     curl     && rm -rf /var/lib/apt/lists/*:
0.716 Get:1 https://packages.microsoft.com/debian/11/prod bullseye InRelease [3650 B]
0.765 Err:1 https://packages.microsoft.com/debian/11/prod bullseye InRelease
0.765   At least one invalid signature was encountered.
0.988 Get:2 http://deb.debian.org/debian bullseye InRelease [75.1 kB]
1.145 Err:2 http://deb.debian.org/debian bullseye InRelease
1.145   At least one invalid signature was encountered.
1.221 Get:3 http://deb.debian.org/debian-security bullseye-security InRelease [27.2 kB]
1.254 Err:3 http://deb.debian.org/debian-security bullseye-security InRelease
1.254   At least one invalid signature was encountered.
1.336 Get:4 http://deb.debian.org/debian bullseye-updates InRelease [44.0 kB]
1.376 Err:4 http://deb.debian.org/debian bullseye-updates InRelease
1.376   At least one invalid signature was encountered.
1.560 Get:5 https://archive.mariadb.org/mariadb-10.11/repo/debian bullseye InRelease [4634 B]
1.588 Err:5 https://archive.mariadb.org/mariadb-10.11/repo/debian bullseye InRelease
1.588   At least one invalid signature was encountered.
2.123 Get:6 https://apt.postgresql.org/pub/repos/apt bullseye-pgdg InRelease [107 kB]
2.395 Err:6 https://apt.postgresql.org/pub/repos/apt bullseye-pgdg InRelease
2.395   At least one invalid signature was encountered.
2.403 Reading package lists...
2.413 W: GPG error: https://packages.microsoft.com/debian/11/prod bullseye InRelease: At least one invalid signature was encountered.
2.413 E: The repository 'https://packages.microsoft.com/debian/11/prod bullseye InRelease' is not signed.
2.413 W: GPG error: http://deb.debian.org/debian bullseye InRelease: At least one invalid signature was encountered.
2.413 E: The repository 'http://deb.debian.org/debian bullseye InRelease' is not signed.
2.413 W: GPG error: http://deb.debian.org/debian-security bullseye-security InRelease: At least one invalid signature was encountered.
2.413 E: The repository 'http://deb.debian.org/debian-security bullseye-security InRelease' is not signed.
2.413 W: GPG error: http://deb.debian.org/debian bullseye-updates InRelease: At least one invalid signature was encountered.
2.413 E: The repository 'http://deb.debian.org/debian bullseye-updates InRelease' is not signed.
2.413 W: GPG error: https://archive.mariadb.org/mariadb-10.11/repo/debian bullseye InRelease: At least one invalid signature was encountered.
2.413 E: The repository 'https://archive.mariadb.org/mariadb-10.11/repo/debian bullseye InRelease' is not signed.
2.413 W: GPG error: https://apt.postgresql.org/pub/repos/apt bullseye-pgdg InRelease: At least one invalid signature was encountered.
2.413 E: The repository 'https://apt.postgresql.org/pub/repos/apt bullseye-pgdg InRelease' is not signed.
------

 1 warning found (use docker --debug to expand):
 - UndefinedVar: Usage of undefined variable '$PYTHONPATH' (line 70)
Dockerfile:11
--------------------
  10 |     # Install system dependencies for data science packages
  11 | >>> RUN apt-get update && apt-get install -y \
  12 | >>>     gcc \
  13 | >>>     g++ \
  14 | >>>     make \
  15 | >>>     pkg-config \
  16 | >>>     libfreetype6-dev \
  17 | >>>     libpng-dev \
  18 | >>>     libfontconfig1-dev \
  19 | >>>     curl \
  20 | >>>     && rm -rf /var/lib/apt/lists/*
  21 |     
--------------------
ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -o errexit -o nounset -o nolog -c apt-get update && apt-get install -y     gcc     g++     make     pkg-config     libfreetype6-dev     libpng-dev     libfontconfig1-dev     curl     && rm -rf /var/lib/apt/lists/*" did not complete successfully: exit code: 100
ERROR: Service 'airflow-standalone' failed to build : Build failed
(base) admin2@labvm:~/Desktop/stellar-mlops$ git init
error: failed to write new configuration file /home/admin2/Desktop/stellar-mlops/.git/config.lock
fatal: could not set 'core.repositoryformatvers
