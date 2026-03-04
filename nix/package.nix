# nix/package.nix
#
# Builds the open-amplify-ai Python package from source using the uv_build backend.
#
# Usage (from a NixOS configuration or another nix expression):
#
#   pkgs.callPackage ./nix/package.nix {}
#
{ lib
, python3Packages
, fetchgit
}:

python3Packages.buildPythonPackage rec {
  pname = "open-amplify-ai";
  version = "0.0.1";
  pyproject = true;

  # Build from the project root (two levels up from this file).
  # When imported via callPackage the src must point to the repository root.
  src = ./..;

  nativeBuildInputs = [
    python3Packages.uv-build
  ];

  propagatedBuildInputs = with python3Packages; [
    python-dotenv
    requests
    fastapi
    uvicorn
    tqdm
    pypdf
    python-multipart
  ];

  # The package ships a console_scripts entry point "amplify".
  # Tests require network access (Amplify AI API), so skip them in the
  # Nix sandbox.
  doCheck = false;

  meta = with lib; {
    description = "OpenAI-compatible wrapper for the Amplify AI API";
    homepage    = "https://github.com/loeeeee/amplify-ai";
    license     = licenses.mit;
    maintainers = [];
    mainProgram = "amplify";
  };
}
