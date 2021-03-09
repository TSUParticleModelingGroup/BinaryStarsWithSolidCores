# BinaryStarsWithSolidCores
This code will generate a binary start system. The two stars will have solid core and hydrogen plasma main bodies. The main body of the stars will be able to grow so we can study the action of contact binary stars.

To compile the code first you will need to make the compile script exicutable. Type 

chmod 777 ./compile 

in the command line of a terminal that contains the files.
Then type ./compile on a command line in this folder and the compile script will compile all the source code.

To generate a set of starts first open the BuildSetup file and sellect the comfiguration you want. Then type ./BuildStars in comand line. It will generate 2 stars and put them in a new star folder that is time stamped.

To generate a simulation go into the star folder you just create and set the BranchSetup file with the initial conditions you want then type ./BranchRun. It will generate A branch fold for you that is time stamped.

If you want to add time to your simulation go into your newly created branch folder and type ./ContinueRun after fitting inter it will prompt for the time you would like to add to the run.  
