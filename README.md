How to navigate through the project files:
- Opening the Project3 folder
- For source codes, you can go through `Project3/Project3/` of this repository [ImageStitchingProjectFiles](https://github.com/akashrai2003/ImageStitchingProjectFiles/tree/master). The size of the project was large and couldn't be incorporated inside the same repository. Find the `.cpp` files there, in which `CAF.cpp` has the whole code of image stitching while inpainting, and `FileName.cpp` has the CLI of our application. We further plan to incorporate a GUI too.
- For the results of the code, go into `Project3/x64/Release/` and find the previously stored outputs inside the `/DesiredOutput` folder for your reference. After building the project from source again, you can run the executable file with the command after going into the `./x64/Release` folder: 
  ```bash
  .\Project3.exe img3.jpg img1.jpg img4.jpg img2.jpg --features orb --matcher homography --save_graph match.dot
