# nix/module.nix
#
# NixOS module for the amplify-ai OpenAI-compatible server.
#
# Usage in configuration.nix:
#
#   imports = [ /path/to/amplify-ai/nix/module.nix ];
#
#   services.amplify-ai = {
#     enable          = true;
#     environmentFile = /run/secrets/amplify-ai.env;
#   };
#
# The environmentFile must contain (one per line, no extra spaces):
#   AMPLIFY_AI_TOKEN=amp-v1-<your-token>
#   AMPLIFY_AI_EMAIL=you@vanderbilt.edu
#
# SECURITY: Never place secrets inside the Nix store. Use a secrets manager
# (sops-nix, agenix, etc.) or manually create the file with mode 0400 owned
# by root before running nixos-rebuild.
#
{ config, lib, pkgs, ... }:

let
  cfg = config.services.amplify-ai;

  # Build the package from the repository root so the module is self-contained.
  amplifyPackage = pkgs.callPackage ./package.nix {};

in {
  options.services.amplify-ai = {

    enable = lib.mkEnableOption "amplify-ai OpenAI-compatible API server";

    host = lib.mkOption {
      type        = lib.types.str;
      default     = "127.0.0.1";
      description = ''
        Address the server binds to.
        Use 0.0.0.0 to expose on all interfaces (combine with a firewall rule
        or a reverse proxy).
      '';
    };

    port = lib.mkOption {
      type        = lib.types.port;
      default     = 8000;
      description = "TCP port the server listens on.";
    };

    environmentFile = lib.mkOption {
      type        = lib.types.path;
      description = ''
        Path to a file containing the runtime secrets for the service.
        The file is loaded by systemd at service start and must contain:
          AMPLIFY_AI_TOKEN=amp-v1-...
          AMPLIFY_AI_EMAIL=you@vanderbilt.edu
        The file must NOT be world-readable (chmod 400 is recommended).
      '';
    };

    dataDir = lib.mkOption {
      type        = lib.types.path;
      default     = "/var/lib/amplify-ai";
      description = "Directory used for logs and runtime state.";
    };

    openFirewall = lib.mkOption {
      type        = lib.types.bool;
      default     = false;
      description = "Whether to open the firewall for the configured port.";
    };

  };

  config = lib.mkIf cfg.enable {

    # Open the firewall port when requested.
    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    systemd.services.amplify-ai = {
      description = "Amplify AI OpenAI-compatible API server";
      documentation = [ "https://github.com/loeeeee/amplify-ai" ];

      after    = [ "network-online.target" ];
      wants    = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        # DynamicUser allocates a transient UID/GID automatically; no need to
        # create a dedicated system user.
        DynamicUser = true;

        # Persist the log directory across restarts. systemd maps
        # StateDirectory to /var/lib/<name> and sets $STATE_DIRECTORY.
        StateDirectory = "amplify-ai";
        WorkingDirectory = cfg.dataDir;

        # Secrets are injected from the user-supplied file at runtime.
        # The file is read by systemd and its contents are NOT placed in the
        # Nix store or the service unit (which is world-readable).
        EnvironmentFile = cfg.environmentFile;

        # Bind host/port via environment so server.py can pick them up.
        Environment = [
          "AMPLIFY_SERVER_HOST=${cfg.host}"
          "AMPLIFY_SERVER_PORT=${toString cfg.port}"
        ];

        ExecStart = "${amplifyPackage}/bin/amplify server";

        # Restart on failure with a short back-off to survive token expiry
        # or transient network errors.
        Restart    = "on-failure";
        RestartSec = "5s";

        # Systemd hardening — reduce the attack surface.
        PrivateTmp          = true;
        ProtectSystem       = "strict";
        ProtectHome         = true;
        NoNewPrivileges     = true;
        PrivateDevices      = true;
        ProtectKernelModules = true;
        ProtectKernelTunables = true;
        ProtectControlGroups  = true;
        RestrictNamespaces    = true;
        LockPersonality       = true;
        MemoryDenyWriteExecute = true;
        RestrictRealtime      = true;
        RestrictSUIDSGID      = true;
        SystemCallFilter      = "@system-service";
        SystemCallErrorNumber = "EPERM";
      };
    };

  };
}
