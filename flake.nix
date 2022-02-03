{
  description = "Plots how CPython matches regexes";

  inputs = {
   nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils }:  flake-utils.lib.eachDefaultSystem (system: let 
    pkgs = import nixpkgs {
      inherit system;
    };
    railroad-diagrams = pkgs.python310Packages.buildPythonPackage rec {
      pname = "railroad-diagrams";
      version = "1.1.1";

      src = pkgs.python310Packages.fetchPypi {
        inherit pname version;
        sha256 = "sha256-ih7CJ2Zr4gAOdnlKp0D3eYfxWGB3quTQkNJjOzBkyXY=";
      };
    };
  in {
    devShell = pkgs.mkShell {
      name = "cpython-regex-plotter-shell";

      buildInputs = [
        railroad-diagrams
        pkgs.python310Packages.hypothesis
        pkgs.python310Packages.pytest
      ];
    };
  });
}
