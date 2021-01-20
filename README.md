# BinaryStarsWithSolidCores
This code will generate a binary start system. The two stars will have solid core and hydrogen plasma main bodies. The main body of the stars will be able to grow so we can study the action of contact binary stars.

* Compile Examples
  ```bash
  nvcc StarBranchRun.cu -o StarBranchRun.exe -lglut -lGL -lGLU -lm
  nvcc StarBranchRun.cu -o StarBranchRun.exe -lglut -lGL -lGLU -lm --use_fast_math
  ```
