# Official Docker base image

This directory contains the former `E:\EDA官方dockers` workspace.

- `edarace-agent-base_ubuntu24.tar`: the original official image archive.
- `edarace-agent-base_ubuntu24/`: the unpacked OCI image layout.

Both forms are retained intentionally: the tar file is the original importable
artifact, while the OCI directory supports inspection without unpacking it
again.  Large image blobs are excluded from Git and from the project Docker
build context.
