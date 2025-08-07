{
    description = "A character-level recurrent neural network implementation.";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

        pyproject-nix = {
            url = "github:pyproject-nix/pyproject.nix";
            inputs.nixpkgs.follows = "nixpkgs";
        };

        uv2nix = {
            url = "github:pyproject-nix/uv2nix";
            inputs.pyproject-nix.follows = "pyproject-nix";
            inputs.nixpkgs.follows = "nixpkgs";
        };

        pyproject-build-systems = {
            url = "github:pyproject-nix/build-system-pkgs";
            inputs.pyproject-nix.follows = "pyproject-nix";
            inputs.uv2nix.follows = "uv2nix";
            inputs.nixpkgs.follows = "nixpkgs";
        };
    };

    outputs =
        inputs:
        let
            inherit (inputs.nixpkgs) lib;

            workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

            overlay = workspace.mkPyprojectOverlay {
                sourcePreference = "wheel";
            };

            pkgs = inputs.nixpkgs.legacyPackages.x86_64-linux;

            python = pkgs.python313;

            pythonSet =
                (pkgs.callPackage inputs.pyproject-nix.build.packages {
                    inherit python;
                }).overrideScope
                    (
                        lib.composeManyExtensions [
                            inputs.pyproject-build-systems.overlays.default
                            overlay
                        ]
                    );
        in
        {
            packages.x86_64-linux = pythonSet.mkVirtualEnv "char-rnn-env" workspace.deps.default;

            uv2nix =
                let
                    editableOverlay = workspace.mkEditablePyprojectOverlay {
                        root = "$REPO_ROOT";
                    };

                    editablePythonSet = pythonSet.overrideScope (
                        lib.composeManyExtensions [
                            editableOverlay

                            (final: prev: {
                                char-rnn = prev.char-rnn.overrideAttrs (old: {
                                    src = lib.fileset.toSource {
                                        root = old.src;
                                        fileset = lib.fileset.unions [
                                            (old.src + "/pyproject.toml")
                                            (old.src + "/README.md")
                                            (old.src + "/src/char_rnn/__init__.py")
                                        ];
                                    };

                                    nativeBuildInputs =
                                        old.nativeBuildInputs
                                        ++ final.resolveBuildSystem {
                                            editables = [ ];
                                        };
                                });
                            })
                        ]
                    );

                    virtualenv = editablePythonSet.mkVirtualEnv "char-rnn-dev-env" workspace.deps.all;
                in
                pkgs.mkShell {
                    packages = [
                        virtualenv
                        pkgs.uv
                        pkgs.pylint
                        pkgs.yapf
                        pkgs.pyright
                    ];

                    env = {
                        UV_NO_SYNC = "1";
                        UV_PYTHON = python.interpreter;
                        UV_PYTHON_DOWNLOADS = "never";
                    };

                    shellHook = ''
                        unset PYTHONPATH
                        export REPO_ROOT=$(git rev-parse --show-toplevel)
                    '';
                };
        };
}
