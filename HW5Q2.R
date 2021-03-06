# Approximating a compound poisson random variable 

numb_sn <- 100000 # M 
lambda <- 100 # Poisson process parameter 
mu = 0.10 # Parameter for X 
sigma = 0.4 # Parameter for X

# Begin approximating Sn 
# Generate random values of N from poisson distribution 
N_vector <- rpois(numb_sn, lambda)

# Compute a sample of Sn 
Sn_sample <- function(N, mean = mu, std = sigma){
  # Draw N random variables from distribution of X
  log_X_k <- rnorm(N, mean = mu, sd = sigma)
  X_k <- exp(log_X_k)
  Sn <- sum(X_k)
  return (Sn)
}

# Pre allocate space 
Sn_MC_samples <- rep(0, numb_sn)

# Compute M copies of Sn 
for (sample in 1:numb_sn){
  N_i <- N_vector[sample]
  Sn_MC_samples[sample] <- Sn_sample(N_i, mu, sigma)
  
}

# Plot the distribution with a normal density overlaid 
makeHist <- function(x, color = "blue", title = "Histogram"){
  h<-hist(x,breaks = 100 ,main=title) 
  xfit<-seq(min(x),max(x),length=100) 
  yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
  yfit <- yfit*diff(h$mids[1:2])*length(x) 
  lines(xfit, yfit, col=color, lwd=2)
}

makeHist(Sn_MC_samples)

####################### Quantile Stuff #############################
#################################################################

quantiles <- seq(from = 0.95, to = 0.99999, by = 0.00001)

# Empirical distribution quantiles 
empirical_tail <- quantile(Sn_MC_samples, quantiles)
emp_tail <-1- empirical_tail 

# Normal Approximation quantiles 
Sn_mean <- exp(mu)*lambda
Sn_sd <- lambda*sigma^2* + exp(mu^2)*lambda

norm_tail_approx2 <- qnorm(quantiles, Sn_mean, Sn_sd)
norm_tail2 <- 1- norm_tail_approx2 

norm_tail_approx <-qnorm(quantiles, mean(Sn_MC_samples), sd(Sn_MC_samples))
norm_tail <- 1- norm_tail_approx
# Gamma tail approx ... Equations from the book - pg 477 of pdf 
# Or simplified equations on 497 in appendix
EX1 <- mean(Sn_MC_samples) # E[X]
EX2 <- mean(Sn_MC_samples^2) # E[X^2]
EX3 <- mean(Sn_MC_samples^3) # E[X^3]

alpha_g <- (2*sqrt(lambda*EX2^3)/EX3)^2
beta_g <- sqrt(alpha_g/ (lambda*EX2))
k_g <- lambda*EX1 - alpha_g/beta_g

library(FAdist)
gamma_tail_approx2 <- qgamma3(quantiles,shape = alpha_g, scale = beta_g, thres=k_g )
gamma_tail2 <-1- gamma_tail_approx2 


# Find the parameters using MLE in the MASS package
library(MASS)
gamma_params <- fitdistr(Sn_MC_samples, "gamma")
gamma_shape <- gamma_params$estimate[1]
gamma_rate <- gamma_params$estimate[2]
# Compute the quantiles and find the tail distribution 
gamma_tail_approx <- qgamma(quantiles,shape = gamma_shape, rate = gamma_rate )
gamma_tail <- 1- gamma_tail_approx 



# Plot on log log scale 
# NOT ON LOG LOG SCALE
plot(emp_tail, quantiles, type = 'l', col = 'red', lwd = 2)
lines(quantiles, norm_tail, type = 'l', col = 'blue', lwd = 2)
lines(quantiles, gamma_tail, type = 'l', col = 'orange', lwd = 2)
legend("bottomleft", c("Empirical", "Normal", "Gamma"), lwd = 2, col = c("red", "blue", "orange"))

# The approximations should not straddle the empirical distribution 


# Plotting again to look more like his but same analytical problems 
# Switch x and y axis 

plot(emp_tail, quantiles, type = 'l', col = 'red', lwd = 2)
lines(norm_tail, quantiles, type = 'l', col = 'blue', lwd = 2)
lines(gamma_tail, quantiles, type = 'l', col = 'orange', lwd = 2)
legend("bottomleft", c("Empirical", "Normal", "Gamma"), lwd = 2, col = c("red", "blue", "orange"))

# The gamma approximation seems to work better because it is close to the empirical distribution. 