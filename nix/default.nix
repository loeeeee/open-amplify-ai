# nix/default.nix
#
# Convenience entry point that exposes both the package derivation and the
# NixOS module so callers can import a single path.
#
# Classic NixOS usage:
#
#   let
#     amplifyAi = import /path/to/amplify-ai/nix { inherit pkgs; };
#   in {
#     imports           = [ amplifyAi.module ];
#     services.amplify-ai.enable = true;
#     # ... other options ...
#   }
#
{ pkgs ? import <nixpkgs> {} }:

{
  package = pkgs.callPackage ./package.nix {};
  module  = ./module.nix;
}
