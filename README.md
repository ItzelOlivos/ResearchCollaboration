# ResearchCollaboration

This project is a web scrapper that collects data from Google Scholar and returns a network of collaboration.

============= To generate a new collaboration network ======================================= 
1. Edit the file people.txt (here you can add the google scholar users of people working in a field of your interest)
2. Type:
  
  > python main.py --t c --p people.txt --e my_depth --o my_title

Notes: 
* The variable my_depth indicates the number of hops the scrapper can take. Eg, my_depth = 1 will collect the profiles of the collaborators of the users listed in people.txt and then will explore the profiles of the collaborators of these last profiles. WARNING: my_depth = 2 might be too ambitious, collaborations grow exponentially!
* The variable my_title indicates the title of the files that the scrapper will generate.

 
============= To just generate the (sub)graph of data already collected type =================

> python main.py --t d --p people.txt --e 1 --o my_title

Then enter the percentage of nodes that you want to display. Open the file minds.html to visualize the network.
