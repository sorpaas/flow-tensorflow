# Flow's Tensorflow Component

*This is work-in-progress. The program may be buggy. You are warned!*

This is intended to be used with [Flow](https://github.com/sorpaas/flow). It
parses Flow's graph JSON representation and uses tensorflow to train it
accordingly. The whole program is exposed as a HTTP API.

## Quickstart

### Install Tensorflow

If you use [Nix](http://nixos.org/nix/), simply run `nix-shell` in the project's
root. Otherwise refer to
[Tensorflow's documentation](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html).

### Start the Server

With python install, run:

```bash
python serve.py
```
