How to navigate through the project files: 
        - Opening the Project3 folder 
        - For source codes you can go through Project3/Project3/ of this repository https://github.com/akashrai2003/ImageStitchingProjectFiles/tree/master as the size of project was large and couldn't be incorporated inside the same repository, and find the .cpp files there in which CAF.cpp has the whole code of image stitching while inpainting and the FileName.cpp has the CLI of our application and we further will incorporate a GUI too.
        - For the results of code you can go into Project3/x64/Release/  and you can see the previously stored outputs inside /DesiredOutput folder for your reference and then after building the project from source again you can run the executable file with the command after going into the ./x64/Release folder: 
        .\Project3.exe img3.jpg img1.jpg img4.jpg img2.jpg --features orb --matcher homography --save_graph match.dot 
        - To run the executable files directly you can redirect to the executables directory and run the CLI from the /GUI folder or you can open up a terminal and go to the /StaticLinkedExecutable and directly run the above command in your terminal specifying the image paths correctly.
        - Setting up OpenCV for static linking in Visual Studio 17 2022 has been provided in the docx file. 