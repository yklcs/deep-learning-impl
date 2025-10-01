{
  description = "CUDA development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          cudaSupport = true;
          allowUnfree = true;
        };
      };
      cudaArchitectures = "90";
      packages =
        with pkgs;
        [
          ninja
          ffmpeg
          cmake
          openssl
          jq
        ]
        ++ (with cudaPackages_12_8; [
          cuda_cccl
          cuda_cudart
          cuda_cupti
          cuda_nvcc
          cuda_nvml_dev
          cuda_nvrtc
          cuda_nvtx
          cusparselt
          cutensor
          cudnn
          libcublas
          libcufft
          libcufile
          libcurand
          libcusolver
          libcusparse
        ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        inherit packages;
        shellHook =
          let
            libPath = with pkgs; lib.makeLibraryPath (packages ++ [ addDriverRunpath.driverLink ]);
          in
          ''
            export CMAKE_CUDA_ARCHITECTURES=${cudaArchitectures}
            export LD_LIBRARY_PATH=${libPath}:$NIX_LD_LIBRARY_PATH
            export TRITON_LIBCUDA_PATH=${libPath}:$NIX_LD_LIBRARY_PATH
          '';
      };
      formatter.${system} = nixpkgs.legacyPackages.${system}.nixpkgs-fmt;
    };
}
