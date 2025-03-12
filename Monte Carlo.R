
## load packages and functions 

source("C:\\Users\\aless\\OneDrive\\Desktop\\Vienna\\Upload\\functions_packages v2.R")


### store all available info 

c0 = 45000 ## initial investment
time.period = 5 ## investment lifetime
depr.yearly = c0/time.period ## depreciation constant
fixed.cost = 4000 ## annual fixed costs
tax.rate = 0.4 ## tax rate 
inv = c(-3000, -1500, 0, 1700, 2800) ## change in working capital
revenues.t1 = 60000 ## revenues t1
revenues.grate = 0.03 ## revenues growth rate
var.costs.t1 = 33000 ## variable cost
var.costs.grate = 0.025 ## variable cost growth rate
k = 0.11 ## annual discount rate

### financial model 

## revenues t1-t5
revenues = c(revenues.t1, revenues.t1*(1+revenues.grate)^(1:(time.period-1)))
## variable costs t1-t5
var.costs = c(var.costs.t1, var.costs.t1*(1+var.costs.grate)^(1:(time.period-1)))
## fixed costs t1-t5
fixed.costs = rep(fixed.cost, time.period)
# ebitda
ebitda = revenues-var.costs-fixed.costs
## deprec
deprecetation = rep(depr.yearly, time.period)
## ebit 
ebit = ebitda-deprecetation
## taxes
taxes = ebit*tax.rate
## nopat
nopat = ebit-taxes
## ebidta net of taxes 
ebidta.net = nopat+deprecetation
## free cash flow 
free.cf = ebidta.net+inv

## cashflow t0-t5
cashflows = c(-c0, free.cf)

### compute npv (discount rate 0.11) 
npv.res = npv(cashflows, k = 0.11) 
cat(paste0("NPV: ", round(npv.res, 2)))

### compute internal rate of return
irr.res = irr(cashflows)
cat(paste0("IRR: ", round(irr.res*100, 2), "%"))





#### Monte Carlo Simulation 

## wrap the financial model within a function: 
## having as inputs the variables that we consider
## stochastic and outputting the cahsflows 

compute_cf = function(c0, revenues.t1, revenues.grate, var.costs.t1, var.costs.grate){

time.period = 5 ## investment lifetime
depr.yearly = c0/time.period ## deprecetation constant
fixed.cost = 4000 ## annual fixed costs
tax.rate = 0.4 ## tax rate 
inv = c(-3000, -1500, 0, 1700, 2800) ## change in working capital

### financial model 

## revenues t1-t5
revenues = c(revenues.t1, revenues.t1*(1+revenues.grate)^(1:(time.period-1)))
## variable costs t1-t5
var.costs = c(var.costs.t1, var.costs.t1*(1+var.costs.grate)^(1:(time.period-1)))
## fixed costs t1-t5
fixed.costs = rep(fixed.cost, time.period)
# ebitda
ebitda = revenues-var.costs-fixed.costs
## deprec
deprecetation = rep(depr.yearly, time.period)
## ebit 
ebit = ebitda-deprecetation
## taxes
taxes = ebit*tax.rate
## nopat
nopat = ebit-taxes
## ebidta net of taxes 
ebidta.net = nopat+deprecetation
## free cash flow 
free.cf = ebidta.net+inv

## cashflow t0-t5
cashflows = c(-c0, free.cf)

return(cashflows)

}


## checking convergence 


## number of iterations per replication: 10000
N=1e4
## number of replications
nrep = 3

## initializing the matrix where to store 
## the simulated irr
IRR.mat = matrix(data = NA, nrow = N, ncol = nrep)
colnames(IRR.mat) = c("Sim1", "Sim2", "Sim3")

## initializing the matrix where to store 
## the simulated npv
NPV.mat = matrix(data = NA, nrow = N, ncol = nrep)
colnames(NPV.mat) = c("Sim1", "Sim2", "Sim3")

### outer loop: looping over the columns of the matrix
## ("replication j")
for(j in 1:nrep){
  
  
  ### outer loop: looping over the rows of the matrix
  ## ("iteration i in replication j")
  
  for (i in 1:N) {
    
    ## extract initial investment from triangular distribution
    c0 = rtri(1, min = 43550, max = 46250, mode = 45000)
    ## extract revenues in t1 from uniform distribution
    revenues.t1 = runif(1, 58000, 62300)
    ## extract revenues growth rate from:
    ## mixture between binomial and uniform distribution
    revenues.grate = ifelse(rbinom(1, 1, prob = c(0.75, 0.25))==1, 
                            runif(1, min = 0, max = 0.05), 
                            runif(1, min = -0.08, max = -0.05))
    ## extract initial investment from normal distribution
    var.costs.t1 = rnorm(1, mean = 33000, sd = 500)
    ## extract initial investment from beta distribution
    var.costs.grate = rbeta(1, shape1 = 6, shape2 = 3)*0.055
    ## extract initial investment from lognormal distribution
    k = log(rlnorm(1, meanlog = 0.11, sdlog = 0.005))
    
    ## compute the cashflows
    cf = compute_cf(c0, revenues.t1, revenues.grate, 
                    var.costs.t1, var.costs.grate)
    
    ## compute NPV and IRR
    NPV = npv(cf, k)
    IRR = irr(cf)
    
    ## store the results in the matrix (row i, column j)
    IRR.mat[i, j] = IRR
    NPV.mat[i, j] = NPV
    
  }
  
}

### reshape the matrix from wide to long format
IRR.mat.long = IRR.mat %>%
  as.data.frame() %>% 
  pivot_longer(cols = everything(), 
               names_to = "Simulation", 
               values_to = "IRR")

### plot the grouped boxplots to check if the 
## distribution of the IRR differ across the replications
ggplot(IRR.mat.long, aes(x=IRR, group = Simulation, fill = Simulation))+
  geom_boxplot()+
  coord_flip()+
  theme_minimal()


###############


## run the Monte Carlo simulation


## initialize vectors where to store
## simulated irr and npv values 
irr.temp = c()
npv.temp = c()

for (i in 1:N) {
  
  ## extract initial investment from triangular distribution
  c0 = rtri(1, min = 43550, max = 46250, mode = 45000)
  ## extract revenues in t1 from uniform distribution
  revenues.t1 = runif(1, 58000, 62300)
  ## extract revenues growth rate from:
  ## mixture between binomial and uniform distribution
  revenues.grate = ifelse(rbinom(1, 1, prob = c(0.75, 0.25))==1, 
                          runif(1, min = 0, max = 0.05), 
                          runif(1, min = -0.08, max = -0.05))
  ## extract initial investment from normal distribution
  var.costs.t1 = rnorm(1, mean = 33000, sd = 500)
  ## extract initial investment from beta distribution
  var.costs.grate = rbeta(1, shape1 = 6, shape2 = 3)*0.055
  ## extract initial investment from lognormal distribution
  k = log(rlnorm(1, meanlog = 0.11, sdlog = 0.005))
  
  ## compute the cashflows
  cf = compute_cf(c0, revenues.t1, revenues.grate, 
                  var.costs.t1, var.costs.grate)
  
  ## compute NPV and IRR
  NPV = npv(cf, k)
  IRR = irr(cf)
  
  ## append the values to the initialized vectors
  irr.temp = c(irr.temp, IRR)
  npv.temp = c(npv.temp, NPV)
  
}


## create a dataframe storing the two vectors
irr.df = data.frame(IRR = irr.temp, 
                    NPV = npv.temp)

## plot the histogram of the irr
ggplot(irr.df, aes(x=IRR, y=after_stat(density)))+
  geom_histogram()+
  theme_minimal()


## compute summary stats of the simulated IRR
summary.irr = c(mean(irr.df$IRR), sd(irr.df$IRR), median(irr.df$IRR), 
  quantile(irr.df$IRR, 0.25), quantile(irr.df$IRR, 0.75), 
  min(irr.df$IRR), max(irr.df$IRR))
names(summary.irr) = c("Mean", "SD", "Median", 
                       "Q1", "Q3", "Min", "Max")

summary.irr


## plot the empirical cumulative distribution function of the IRR
ggplot(irr.df, aes(x=IRR))+
  stat_ecdf(geom="step")+
  theme_minimal()

## extract the 5th quantile 
quantile(irr.df$IRR, 0.05)

## extract proportion of values less than or equal to IRR of 0.15
ecdf(irr.df$IRR)(0.15)
