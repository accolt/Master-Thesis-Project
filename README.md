# Master-Thesis-Project
The final paper is performed as a project for one of the biggest mining companies in Austria.
### Description of the project
Detailed information about movements, timestamps, statuses and other conditions of a dump truck is generated every 2 seconds and by the end of each day formed into a csv file on a central computer.
The main task is the optimization of a traffic flow control of vehicles in the mine. A solution on how to improve current truck dispatch system has to be developed.
### Solution
The current dispatch system in the mine was improved by a special Python script. This script works on a data analysis. 
Principle of operation:
* The script is located in a folder on a computer
* A csv file with raw dataset has to be placed into that folder
* Script starts to work by double clicking on it
* After 10-15 seconds in the same folder there will appear 4 png documents with plots and graphs and 1 txt file with idling report
### Achiemenets in programming
By optimising the code it was able to decrease the execution time from 5 hours to 12 seconds. Here are the solutions:
* Introduction of functions that helped to decrease a decent amount of code;
* In the penultimate version of the code, there was a part of enumeration of coordinates. The problem was that there are over 50k 'x' coordinates as well 'y' in each dataset. The initial idea was to form pairs of unique coordinates. To do so a loop inside a loop was created which enormously slowed down the calculation process.
* The final version of code includes splitings rows of 'y' coordintaes into blocks based on a condition. The 'x' coordinates were grouped by a mean value regarding to each 'y' coordinate. This solution turned to be a crucial in speeding up the code execution.
