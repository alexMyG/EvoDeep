
import os
list_individuals_files = []

all_files = os.listdir(".")

list_individuals_files = filter(lambda k: "individuals_list" in k, all_files)



values_to_extract = ["Prob cross", "Prob mut"]

list_accuracy = "execution_id," + ','.join(values_to_extract) + ","

for index, individual_file in enumerate(list_individuals_files):
    print "FILE: " + individual_file
    file_open = open(individual_file, "r")
    file_lines = file_open.readlines()
    file_id = individual_file.replace("individuals_list_", "").replace(".txt", "")
    
    list_values_individuals_file = []
    for value_to_extract in values_to_extract:
            
        
        for line in file_lines:
            line = line.replace("\n", "")
            if line.startswith(value_to_extract):
                list_values_individuals_file.append(line.split(": ")[1])
                break
                
                
    file_open.close()
    
    file_accuracy_open = open("accuracy_list_" + file_id + ".csv")
    file_accuracy_lines = file_accuracy_open.readlines()
    if index == 0:
        head_line = file_accuracy_lines[0].replace("\n", "").replace(" ", "")
        list_accuracy += head_line + "\n"
    print "extracting accuracy lines: " + str(len(file_accuracy_lines))
    for line in file_accuracy_lines[1:]:
        #print ",".join(list_values_individuals_file) + "," + line
        list_accuracy += (file_id + "," + ",".join(list_values_individuals_file) + "," + line)
    
    file_accuracy_open.close()
    
output_file_accuracy_parameters = "output_file_accuracy_parameters20ex.csv"

file_output_open = open(output_file_accuracy_parameters, "w")

file_output_open.write(list_accuracy)
    
file_output_open.close()
        
    
    
    

