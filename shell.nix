{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python313
    pkgs.uv
  ];

  shellHook = ''
    # Create the virtual environment if it doesn't exist
    if [ ! -d .venv ]; then
      echo "Initializing virtual environment..."
      uv venv
    fi

    # Activate the virtual environment automatically
    source .venv/bin/activate
  '';
}
