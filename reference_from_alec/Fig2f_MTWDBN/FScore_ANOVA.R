# Load necessary libraries
library(ez)
library(readr)

# Read the data from the CSV file
df <- read_csv("Documents/MTWDBN/Final_Figures/Synthetic/FScores.csv")

# Convert factors
df$Model <- as.factor(df$Model)
df$Drop <- as.factor(df$Drop)
df$Iteration <- as.factor(df$Iteration)

# Run the mixed-design ANOVA
ezANOVA(data = df, 
        dv = FScore, 
        wid = Iteration, 
        within = .(Model), 
        between = .(Drop),
        type = 3)  # Type III sum of squares


##############################################################
#Run the post-hoc Tukey Test for each drop level individually#
##############################################################


# Create a list of unique drop levels
drop_levels <- unique(df$Drop)

# Initialize an empty list to store the results
results <- list()

# Loop through each drop level
for (i in 1:length(drop_levels)) {
  
  # Subset the data for the current drop level
  df_sub <- df[df$Drop == drop_levels[i],]
  
  # Fit the linear mixed effects model
  lme.model <- lme(FScore ~ Model, random = ~1|Iteration, data = df_sub)
  
  # Perform the pairwise comparisons and store the results
  results[[i]] <- summary(glht(lme.model, mcp(Model = "Tukey")), test = adjusted("bonferroni"))
  
  # Print the drop level and the results
  print(paste("Drop Level:", drop_levels[i]))
  print(results[[i]])
}

