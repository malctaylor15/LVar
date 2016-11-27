# Homework 5 Part 2 

# Read data and look at summary statistics
data <- read.csv("SP500_Log_Returns_20150101_20160101.csv")
head(data)
Returns <- data[[4]]
plot(Returns, type = 'l')
hist(Returns, breaks = 30)

# Initialize parameters
V <- 1e6
alpha <- 0.95
beta <- 0.025
numb_sim <- 100000 # M 
n <- length(Returns) # number of observations


# Part a 
Return_mean <- mean(Returns)
Return_var <- var(Returns)
norm_inverse <- qnorm((1-alpha), mean= 0, sd=1)
Var_known = -V*(1 - exp(Return_mean +sqrt(Return_var)* norm_inverse))

# Use slide 27 equations for upper and lower sigma paramter 
conf_int_quantile_down<- beta/2
conf_int_quantile_up <- 1-beta/2
conf_ints <- c(conf_int_quantile_down, conf_int_quantile_up)
chisq_df <- n-1

sigmas2_known <- ((n-1)*Return_var)/qchisq(p = conf_ints , df = chisq_df)
#sigma_down <- ((n-1)*Return_var)/qchisq(p = conf_int_quantile_down, df = chisq_df) 

# Watch the signs?
Var_known_conf_int <- -V*(1 - exp(Return_mean +sqrt(sigmas2_known)* norm_inverse))
# Var_down <- -V*(1 - exp(Return_mean +sqrt(sigma_down)* norm_inverse))



############################################
############# Part b ########################
###########################################


# Following steps on slide 31 

# First simulate sigma 

chisq_MC <- rchisq(numb_sim, chisq_df)
sigma2_MC <- ((n-1)*Return_var)/(chisq_MC)
sigma_MC <- sqrt(sigma2_MC)

# Simulate mean of the VaR 
for(sim in 1:numb_sim){
  mu_MC <- rnorm(1,mean = Return_mean, sd = (sigma_MC[sim]/sqrt(n)))
}

# Simulate VaR alpha 
VaR_alpha_MC = -V*(1 - exp(mu_MC +sigma_MC* norm_inverse))
hist(VaR_alpha_MC, breaks = 40)
VaR_MC_mean <- mean(VaR_alpha_MC)
VaR_MC_conf_int <- quantile(VaR_alpha_MC,conf_ints)

# Compare two method of finding VaR

Var_known
Var_known_conf_int

VaR_MC_mean
VaR_MC_conf_int

# The two methods are very similar. The means are slightly different but the confidence intervals overlap for the majority of the time. 
