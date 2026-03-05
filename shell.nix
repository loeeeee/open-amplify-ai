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
    
    # Add a command to easily start the server locally
    start-server() {
      uv run amplify server "$@"
    }
    
    # Print helpful commands to the console
    echo ""
    echo "--- Amplify AI Compatibility Layer ---"
    echo "  Environment is active."
    echo "  Run 'start-server' to start the local testing server."
    echo "--------------------------------------"
  '';
}
