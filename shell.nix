{ pkgs ? import <nixpkgs> {} }:

with pkgs;

stdenv.mkDerivation {
  name = "jupyter-tensorflow-tutorial-env";

  buildInputs = [
    inkscape
    pandoc
    python27
    python27Packages.tensorflow
    python27Packages.jupyter
    python27Packages.flask
  ];

  SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";
}
