# ImageStitchingCPP-BoschProject

How to navigate through the project files:
- Opening the Project3 folder
- For source codes, you can go through `Project3/Project3/` of this repository [ImageStitchingProjectFiles](https://github.com/akashrai2003/ImageStitchingProjectFiles/tree/master). The size of the project was large and couldn't be incorporated inside the same repository. Find the `.cpp` files there, in which `CAF.cpp` has the whole code of image stitching while inpainting, and `FileName.cpp` has the CLI of our application. We further plan to incorporate a GUI too.
- For the results of the code, go into `Project3/x64/Release/` and find the previously stored outputs inside the `/DesiredOutput` folder for your reference. After building the project from source again, you can run the executable file with the command after going into the `./x64/Release` folder: 
  ```bash
  .\Project3.exe img3.jpg img1.jpg img4.jpg img2.jpg --features orb --matcher homography --save_graph match.dot
- The executables folder contains the `.exe` files to directly run the application without any dependencies. So you can head to GUI and run the `project3.exe` file to access the CLI interface where you can enter the number of images you want to stitch and thus enter their paths and finally get the output saved to your directory. Here we have used defaults only to stitch the images 
- In the StaticLinkedExecutable we can run it on a terminal and adding the images to the same directly and run the above given command having more command to be diverse. We had to stick onto ORB features as they were acting the best on the set of our images and thus any other features might be disabled at the moment.
- The word `BoschAutovisionX_1Neuron.docx` file completely explanins our approach and also has a very important setup guideline to static linking of OpenCV in Visual Studio 17 2022.

