library(ggplot2)
library(reshape2)
library(Rmisc)
data <- read.csv("output_file_accuracy_parameters20ex.csv")

data <- melt(data, id.vars = c("execution_id", "Prob.cross", "Prob.mut"))

colnames(data) <- c("execution_id", "Prob.cross", "Prob.mut", "Data portion", "Value")

levels(data$`Data portion`) <- c("Training", "Validation", "Test")
 
data_prob_cross <- data[!data$Prob.mut == 0.3 & !data$Prob.mut == 0.2 & !data$Prob.mut == 0.1,]
data_prob_mut <- data[!data$Prob.cross == 0.3 & !data$Prob.cross == 0.2 & !data$Prob.cross == 0.1,]

tgc_cross <- summarySE(data_prob_cross, measurevar="Value", groupvars=c("Prob.cross","Prob.mut", "`Data portion`"))
tgc_mut <- summarySE(data_prob_mut, measurevar="Value", groupvars=c("Prob.cross","Prob.mut", "`Data portion`"))


#ggplot(data, aes(x=Prob.cross, y=Value, colour = `Data portion`)) + stat_summary(fun.y="mean", geom="line") + 
#  stat_summary(fun.y="mean", geom="point") + labs(x="Cross probability", y="Accuracy") + 
#  scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1))


limits <- aes(ymax = Value + se, ymin=Value - se)


# Error bars
ggplot(tgc_cross, aes(x=Prob.cross, y=Value, colour = `Data portion`)) + geom_errorbar(limits, width=0.02, position=position_dodge(0.03)) + 
  geom_point(position=position_dodge(0.03)) + geom_line(position=position_dodge(0.03)) + 
  labs(x="Cross probability", y="Accuracy") + scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1))
ggsave("CrossProbabilityAccuracy.pdf", width = 5, height = 3.5) 

# Boxplot

data_prob_cross$Prob.cross <- as.factor(data_prob_cross$Prob.cross)
p <- ggplot(data_prob_cross) 
p + geom_boxplot(aes(x=Prob.cross, y=Value, color=`Data portion`)) +  labs(x="Cross probability", y="Accuracy")
ggsave("CrossProbabilityAccuracyBoxplot.pdf", width = 5, height = 3.5) 


# ggplot(data = data_prob_cross , aes(x=Prob.cross, y=Value, colour = `Data portion`)) + geom_boxplot(width = 0.7) 
# 
# ggplot(tgc_cross, aes(x=Prob.cross, y=Value, colour = `Data portion`)) + geom_errorbar(limits, width=0.02, position=position_dodge(0.03)) + 
#   geom_point(position=position_dodge(0.03)) + geom_line(position=position_dodge(0.03)) + 
#   labs(x="Cross probability", y="Accuracy") + scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1))
# ggsave("CrossProbabilityAccuracy.pdf", width = 5, height = 3.5) 
# 



#ggplot(tgc_cross, aes(x=Prob.mut, y=Value, colour = `Data portion`)) + stat_summary(fun.y="mean", geom="line") + 
#  stat_summary(fun.y="mean", geom="point") + labs(x="Mutation probability", y="Accuracy") + 
#  scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1))


# Error bars
ggplot(tgc_mut, aes(x=Prob.mut, y=Value, colour = `Data portion`)) + geom_errorbar(limits, width=0.02, position=position_dodge(0.03)) + 
  geom_point(position=position_dodge(0.03)) + geom_line(position=position_dodge(0.03)) +
  labs(x="Mutation probability", y="Accuracy") + scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1))
ggsave("MutationProbabilityAccuracy.pdf", width = 5, height = 3.5)

# Boxplot

data_prob_mut$Prob.mut <- as.factor(data_prob_mut$Prob.mut)
p <- ggplot(data_prob_mut) 
p + geom_boxplot(aes(x=Prob.mut, y=Value, color=`Data portion`)) + labs(x="Mutation probability", y="Accuracy")
ggsave("MutationProbabilityAccuracyBoxplot.pdf", width = 5, height = 3.5) 










# ALL INDIVIDUALS
data <- read.csv("output_file_accuracy_parameters.csv")
data <- melt(data, id.vars = c("execution_id", "Prob.cross", "Prob.mut"))
colnames(data) <- c("execution_id", "Prob.cross", "Prob.mut", "Data portion", "Value")
data_individuals <- data[data$`Data portion` == "accuracy_test", ]
best_executions <- data_individuals %>% group_by(Prob.cross, Prob.mut, `Data portion`, execution_id) %>% dplyr::summarize(asd =max(Value))
best_executions <- best_executions %>% group_by(Prob.cross, Prob.mut, `Data portion`) %>% dplyr::filter(asd == min(asd))
data_individuals <- data_individuals[data_individuals$execution_id %in% best_executions$execution_id,]
#data_individuals_melt <- melt(data_individuals, id.vars = c("execution_id", "Value"))

data_individuals$grp <- paste(data_individuals$Prob.cross, data_individuals$Prob.mut)

#data_svm_comparison <- data_individuals_melt %>% group_by(variable, value, execution_id) %>% summarise_each(funs(mean), Value)
levels_mut_cross <- levels(as.factor(data_individuals$grp))
data_individuals <- data_individuals[with(data_individuals, order(Value)), ]
data_individuals$x_axis = 0
for (level_mut_cross in levels_mut_cross){

    data_individuals[data_individuals$grp == level_mut_cross,]$x_axis = seq(1,nrow( data_individuals[data_individuals$grp == level_mut_cross,]))
}
# 
# data_individuals_melt[data_individuals_melt$variable == "Prob.cross",]$x_axis <- seq()
#
# data_individuals_melt$x_axis <- row.names(data_individuals_melt)
data_individuals <- data_individuals[data_individuals$x_axis <= 40,]
data_individuals_test <- data_individuals[data_individuals$grp == "0.4 0.2",]
ggplot(data_individuals, aes(x = x_axis, y=Value, colour = grp, linetype = grp)) + geom_line()
ggsave("allIndividuals.pdf", width = 5, height = 5) 
# 
#   geom_point(position=position_dodge(0.03)) + geom_line(position=position_dodge(0.03)) + 
#   labs(x="Cross probability", y="Accuracy") + scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1))
# ggsave("CrossProbabilityAccuracy.pdf", width = 5, height = 3.5) 
# 




# COMPARISON SVM

data_svm_comparison <- read.csv("results_svm_dnn_20ex.csv")

data_svm_comparison <- melt(data_svm_comparison, id.vars = c("execution_id", "Method"))

levels(data_svm_comparison$variable) <- c(levels(data_svm_comparison$variable), "Training") 
levels(data_svm_comparison$variable) <- c(levels(data_svm_comparison$variable), "Test") 

data_svm_comparison$variable[data_svm_comparison$variable == "Accuracy_training"] <- "Training"
data_svm_comparison$variable[data_svm_comparison$variable == "Accuracy_test"] <- "Test"


tgc_svm_comparison <- summarySE(data_svm_comparison, measurevar="value", groupvars=c("Method", "variable"))

tgc_svm_comparison$Method <- factor(tgc_svm_comparison$Method, levels =  c("DNN", "poly", "rbf", "linear", "sigmoid"))

limits <- aes(ymax = value + sd, ymin=value - sd)

ggplot(tgc_svm_comparison, aes(x=Method, y=value, colour = variable)) + geom_errorbar(limits, width=0.2, position=position_dodge(0.03))+
  geom_line(position=position_dodge(0.03), aes(group = variable)) +
  geom_point(position=position_dodge(0.03)) +
  labs(x="Machine learning method", y="Accuracy") + scale_y_continuous(breaks = seq(0,1,0.15), limits = c(0.10, 1.0)) +
  scale_color_discrete(name = "Data portion")
ggsave("ComparisonMLmethods.pdf", width = 5, height = 3) 






















# data_svm <- read.csv("results_svm.csv")
# data_svm_melt <- melt(data_svm,id="Methods") 
# 
# 
# 
# # best value DNN:
# aggregate(data[, 5], list(data[,c(2,3,4)]), mean)
# 
#  #ddply(data,~c(`Data portion`,'Prob.mut'),summarise,mean=mean(Value))
# 
# columns <- names(data)[2:4]
# # Convert character vector to list of symbols
# dots <- lapply(columns, as.symbol)
# 
# data_mean_dnn <- data %>% group_by(Prob.mut,Prob.cross,`Data portion`) %>% summarise_each(funs(mean), Value)
# data_mean_dnn_test <- data_mean_dnn[data_mean_dnn$`Data portion` == "Test",]
# 
# max_accuracy_test <- data_mean_dnn_test[data_mean_dnn_test$Value == max(data_mean_dnn_test$Value),]
# 
# 
# 
# ggplot(data_svm_melt, aes(x=Methods, y=value, fill=factor(variable))) + 
#   stat_summary(fun.y=mean, geom="bar",position=position_dodge(1)) + 
#   scale_color_discrete("variable") +
# stat_summary(fun.data = mean_se,geom="errorbar", color="black",position=position_dodge(1), width=.2)
# 
# 
# 
# 
# 
# ggplot(data_svm, aes(x=Prob.mut, y=Value, colour = `Data portion`)) + stat_summary(fun.y="mean", geom="line") + 
#   stat_summary(fun.y="mean", geom="point") + labs(x="Mutation probability", y="Accuracy") + 
#   scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1)) 
# 
# tgc_svm <- summarySE(data_svm, measurevar="Accuracy_Test", groupvars=c("Methods"))
# 
# # ggplot(data_svm, aes(x=Prob.mut, y=Value, colour = `Data portion`)) + geom_errorbar(limits, width=0.02, position=position_dodge(0.03)) + 
# #   geom_point(position=position_dodge(0.03)) + geom_line(position=position_dodge(0.03)) +
# #   labs(x="Cross probability", y="Accuracy") + scale_y_continuous(breaks = seq(0,1,0.05), limits = c(0.80, 1))
# 
# 
# 
# 
# 
# 
# 
# 

