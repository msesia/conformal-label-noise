options(width = 300)

library(tidyverse)
library(latex2exp)
library(RColorBrewer)

load_data <- function(exp.num) {
    idir <- sprintf("results_hpc/exp%d", exp.num)
    ifile.list <- list.files(idir)
    results <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(), guess_max=2)
    }))


    summary <- results %>%
        pivot_longer(c("Coverage", "Size"), names_to = "Key", values_to = "Value") %>%
        group_by(data, num_var, K, signal, model_name, contamination, epsilon, estimate, n_train, n_cal, Guarantee, Alpha, Label, Method, Key) %>%
        summarise(Mean=mean(Value), N=n(), SE=2*sd(Value)/sqrt(N))

    return(summary)
}

exp.num <- 1
summary <- load_data(exp.num)


method.values = c("Standard", "Adaptive (pessimistic)", "Adaptive (optimistic)")
method.labels = c("Standard", "Adaptive", "Adaptive+")
cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
df.dummy <- tibble(key="Coverage", value=0.95)
df.dummy2 <- tibble(key="Coverage", value=0.5)
color.scale <- cbPalette[c(1,3,7)]
shape.scale <- c(1,2,0)
linetype.scale <- c(1,1,1)

plot.alpha <- 0.1
plot.guarantee <- "lab-cond"
plot.K <- 4
plot.epsilon <- 0.1


# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different values of contamination parameter

make_figure_1 <- function(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(1)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal >= 500)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.85,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s.pdf", 10000, plot.K, plot.guarantee)
        ggsave(file=plot.file, height=3, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_1(plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_1(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)


make_figure_1_slides <- function(plot.alpha=0.1, plot.guarantee="lab-cond", version=1, plot.K=4, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(1)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal >= 500) %>%
        mutate(Method = factor(Method, method.values, method.labels))
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage", "Size","Size"), Mean=c(0.85,1,1.5,4), n_cal=1000, Method="Standard")

    if(version==1) {
        df <- df %>% filter(! Method %in% c("Adaptive", "Adaptive+"))
    } else if (version==2) {
        df <- df %>% filter(! Method %in% c("Adaptive+"))
    } else {
        df <- df
    }
    pp <- df %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="bottom")


    if(save_plots) {
        plot.file <- sprintf("figures_slides/synthetic1_ntrain%d_K%d_%s_%s.pdf", 10000, plot.K, plot.guarantee, version)
        ggsave(file=plot.file, height=4, width=6, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}


make_figure_1_slides(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, version=1, save_plots=TRUE, reload=FALSE)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, version=2, save_plots=TRUE, reload=FALSE)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, version=3, save_plots=TRUE, reload=FALSE)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, version=1, save_plots=TRUE, reload=FALSE)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, version=2, save_plots=TRUE, reload=FALSE)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, version=3, save_plots=TRUE, reload=FALSE)


make_figure_1_lc <- function(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, plot.eps=0.1, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(1)
    }
    label.values <- c(0:(plot.K-1), "marginal")
    label.labels <- c(paste("Label", 1:plot.K, sep=" "), "All labels")

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               epsilon==plot.eps, model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal >= 500)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Label, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_eps%.2f_lc.pdf", 10000, plot.K, plot.guarantee, plot.eps)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}
make_figure_1_lc(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, plot.eps=0.1, save_plots=TRUE, reload=TRUE)

# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the contamination strength, for different calibration set sizes

make_figure_2 <- function(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(2)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(1000,10000,100000))
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(N_cal = sprintf("Cal. samples: %d", n_cal)) %>%
        ggplot(aes(x=epsilon, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~N_cal, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=0, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Strength of label contamination") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_epsilon.pdf", 10000, plot.K, plot.guarantee)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_2(plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, save_plots=TRUE, reload=TRUE)
make_figure_2(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)


# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the contamination strength, for different numbers of classes

make_figure_3 <- function(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(3)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(10000))
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(K_lab = sprintf("%d classes", K)) %>%
        ggplot(aes(x=epsilon, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_wrap(Key~K_lab, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=0, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Strength of label contamination") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_%s_epsilon.pdf", 10000, plot.guarantee)
        ggsave(file=plot.file, height=4, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_3(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=TRUE)
make_figure_3(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=TRUE)




# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different classifiers

make_figure_4 <- function(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, plot.epsilon=0.1, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(4)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, K==plot.K, epsilon==plot.epsilon, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", Alpha==plot.alpha) %>%
        filter(n_cal >= 500)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Model = sprintf("Classifier: %s", model_name)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Model, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_models.pdf", 10000, plot.K, plot.guarantee)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_4(plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, save_plots=TRUE, reload=TRUE)
make_figure_4(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)


# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different values of contamination parameter
# Use different data distributions

make_figure_5 <- function(plot.data, plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(5)
    }

    df <- summary %>%
        filter(data==plot.data, num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal >= 500)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_%s.pdf", 10000, plot.K, plot.guarantee, plot.data)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_5("synthetic2", plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, save_plots=TRUE, reload=TRUE)
make_figure_5("synthetic3", plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, save_plots=TRUE, reload=TRUE)
make_figure_5("synthetic2", plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)
make_figure_5("synthetic3", plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)


# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different values of contamination parameter
# Use different label contamination processes

make_figure_6 <- function(plot.contamination, plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(6)
    }

    df <- summary %>%
        filter(data=="synthetic2", contamination==plot.contamination, num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal >= 500)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_%s.pdf", 10000, plot.K, plot.guarantee, plot.contamination)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_6("block", plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)
make_figure_6("random", plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)


# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different values of contamination parameter
# Use different label contamination processes and try to estimate rho-tilde

make_figure_7 <- function(plot.contamination, plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(7)
    }

    method.values.c = c("Standard-none", "Adaptive (optimistic)-none", "Adaptive (optimistic)-rho")
    method.labels.c = c("Standard", "Adaptive+ (known)", "Adaptive+ (estimated)")
    df.dummy.c <- tibble(key="Coverage", value=0.95)
    df.dummy2.c <- tibble(key="Coverage", value=0.5)
    color.scale.c <- cbPalette[c(1,7,8)]
    shape.scale.c <- c(1,0,0)
    linetype.scale.c <- c(1,1,1)


    df <- summary %>%
        mutate(Method = sprintf("%s-%s", Method, estimate)) %>%
        filter(data=="synthetic2", contamination==plot.contamination, num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal >= 500) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c))


    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_%s_estim-rho.pdf", 10000, plot.K, plot.guarantee, plot.contamination)
        ggsave(file=plot.file, height=3.5, width=7.5, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data(7)
make_figure_7("uniform", plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_7("random", plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_7("uniform", plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_7("random", plot.alpha=0.1, plot.guarantee="marginal", plot.K=4, save_plots=TRUE, reload=FALSE)


# --------------------------------------------------
# Plot calibration-conditional coverage, as a function of calibration set size

load_data_cc <- function(exp.num) {
    idir <- sprintf("results_hpc/exp%d", exp.num)
    ifile.list <- list.files(idir)
    results <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(), guess_max=2)
    }))

    summary <- results %>%
        group_by(data, num_var, K, signal, model_name, contamination, epsilon, estimate, n_train, n_cal, Guarantee, Alpha, Label, Method,
                 calibration_conditional, gamma) %>%
        summarise(Coverage.mean=mean(Coverage), Coverage.10=quantile(Coverage, 0.1), Size=mean(Size)) %>%
        pivot_longer(c("Coverage.mean", "Coverage.10", "Size"), names_to = "Key", values_to = "Value")

    return(summary)
}

make_figure_cc <- function(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_cc(11)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal >= 500)
    df.nominal <- tibble(Key=c("Coverage.10"), Mean=1-plot.alpha) %>%
        mutate(Key = factor(Key, c("Coverage.10","Size"), c("Coverage (10% quantile)","Size (average)")))
    df.range <- tibble(Key=c("Coverage.10","Coverage.10"), Mean=c(0.8,1), n_cal=1000, Method="Standard") %>%
        mutate(Key = factor(Key, c("Coverage.10","Size"), c("Coverage (10% quantile)","Size (average)")))
    pp <- df %>%
        filter(Key %in% c("Coverage.10","Size")) %>%
        mutate(Key = factor(Key, c("Coverage.10","Size"), c("Coverage (10% quantile)","Size (average)"))) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=n_cal, y=Value, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
        scale_x_continuous(trans='log10') +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_cc.pdf", 10000, plot.K, plot.guarantee)
        ggsave(file=plot.file, height=4, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_cc(plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)


# --------------------------------------------------
# Plot calibration-conditional coverage, as a function of calibration set size
# Bounded noise method

load_data_bounded <- function(exp.num) {
    idir <- sprintf("results_hpc/exp%d", exp.num)
    ifile.list <- list.files(idir)
    results <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(), guess_max=2)
    }))


    summary <- results %>%
        pivot_longer(c("Coverage", "Size"), names_to = "Key", values_to = "Value") %>%
        group_by(data, num_var, K, signal, model_name, contamination, epsilon, estimate, n_train, n_cal, Guarantee, Alpha, Label, Method,
                 epsilon_max, epsilon_se, epsilon_alpha, Key) %>%
        summarise(Mean=mean(Value), N=n(), SE=2*sd(Value)/sqrt(N))

    return(summary)
}

make_figure_bounded_1 <- function(plot.alpha=0.1, plot.epsilon=0.1, plot.guarantee="lab-cond", plot.K=4,
                                  plot.epsilon_max=0.1, plot.epsilon_alpha=0.01, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        if(plot.epsilon_max==0.1) {
            summary <- load_data_bounded(31)
        } else if(plot.epsilon_max==0.2) {
            summary <- load_data_bounded(32)
        }
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(1000,10000,100000), epsilon==plot.epsilon, epsilon_max==plot.epsilon_max, epsilon_alpha==plot.epsilon_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        mutate(N_cal = sprintf("Cal. samples: %d", n_cal)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        mutate(epsilon_lower = epsilon - epsilon_se) %>%
        ggplot(aes(x=epsilon_lower, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~N_cal, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
#        scale_x_continuous(trans='log10') +
        xlab("Lower bound for contamination strength") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_K%d_%s_bounded_eps%.2f_emax%.2f_ealpha%.2f.pdf",
                             10000, plot.K, plot.guarantee, plot.epsilon, plot.epsilon_max, plot.epsilon_alpha)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_bounded_1(plot.epsilon=0.1, plot.epsilon_max=0.1, plot.epsilon_alpha=0.01, plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)
make_figure_bounded_1(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.epsilon_alpha=0.01, plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=TRUE)


make_figure_bounded_3 <- function(plot.alpha=0.1, plot.guarantee="lab-cond",
                                  plot.epsilon_max=0.1, plot.epsilon_alpha=0.01, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_bounded(33)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(10000), epsilon==0.2, epsilon_max==plot.epsilon_max, epsilon_alpha==plot.epsilon_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), Method="Standard")
    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        mutate(K_lab = sprintf("%d classes", K)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        mutate(epsilon_lower = epsilon - epsilon_se) %>%
        ggplot(aes(x=epsilon_lower, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_wrap(Key~K_lab, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
#        scale_x_continuous(trans='log10') +
        xlab("Lower bound for contamination strength") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_%s_bounded_emax%.2f_ealpha%.2f.pdf", 10000, plot.guarantee, plot.epsilon_max, plot.epsilon_alpha)
        ggsave(file=plot.file, height=4, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_bounded_3(plot.epsilon_max=0.2, plot.epsilon_alpha=0.01, plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=TRUE)


make_figure_bounded_3_slides <- function(plot.alpha=0.1, plot.guarantee="lab-cond", version=1,
                                  plot.epsilon_max=0.1, plot.epsilon_alpha=0.01, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_bounded(33)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(10000), epsilon==0.2, epsilon_max==plot.epsilon_max, epsilon_alpha==plot.epsilon_alpha) %>%
        mutate(Method = factor(Method, method.values, method.labels))        
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), Method="Standard")

    if(version==1) {
        df <- df %>% filter(! Method %in% c("Adaptive", "Adaptive+"))
    } else if (version==2) {
        df <- df %>% filter(! Method %in% c("Adaptive+"))
    } else {
        df <- df
    }
    
    pp <- df %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        mutate(K_lab = sprintf("%d classes", K)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        mutate(epsilon_lower = epsilon - epsilon_se) %>%
        ggplot(aes(x=epsilon_lower, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_wrap(Key~K_lab, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
#        scale_x_continuous(trans='log10') +
        xlab("Lower bound for contamination strength") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="bottom")

    if(save_plots) {
        plot.file <- sprintf("figures_slides/synthetic1_ntrain%d_%s_bounded_emax%.2f_ealpha%.2f_v%d.pdf",
                             10000, plot.guarantee, plot.epsilon_max, plot.epsilon_alpha, version)
        ggsave(file=plot.file, height=5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data_bounded(33)

make_figure_bounded_3_slides(plot.epsilon_max=0.2, plot.epsilon_alpha=0.01, plot.alpha=0.1, plot.guarantee="lab-cond",
                             version=2, save_plots=TRUE, reload=FALSE)
make_figure_bounded_3_slides(plot.epsilon_max=0.2, plot.epsilon_alpha=0.01, plot.alpha=0.1, plot.guarantee="lab-cond",
                             version=3, save_plots=TRUE, reload=FALSE)


make_figure_bounded_4 <- function(plot.alpha=0.1, plot.guarantee="lab-cond",
                                  save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_bounded(34)
    }

    df <- summary %>%
        filter(data=="synthetic1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(10000), epsilon==0.1, epsilon_alpha==0, epsilon_se==0)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), Method="Standard")
    pp <- df %>%
        filter(epsilon_max <= 1.5* epsilon) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        mutate(K_lab = sprintf("%d classes", K)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=epsilon_max, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_wrap(Key~K_lab, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(trans='log10', breaks=c(1000,2000,5000,10000,20000)) +
#        scale_x_continuous(trans='log10') +
        xlab("Upper bound for contamination strength") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_%s_bounded_emax.pdf", 10000, plot.guarantee)
        ggsave(file=plot.file, height=4, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_bounded_4(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=TRUE)



################
## Estimation ##
################

load_data_estimation <- function(exp.num) {
    idir <- sprintf("results_hpc/exp%d", exp.num)
    ifile.list <- list.files(idir)
    results <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(), guess_max=2)
    }))


    summary <- results %>%
        pivot_longer(c("Coverage", "Size", "epsilon_low", "epsilon_upp", "epsilon_hat"), names_to = "Key", values_to = "Value") %>%
        group_by(data, num_var, K, signal, model_name, contamination, epsilon, estimate, n_train, n_cal, Guarantee, Alpha, Label, Method,
                 epsilon_max, epsilon_alpha, epsilon_train, epsilon_n_clean, epsilon_n_corr, Key) %>%
        summarise(Mean=mean(Value), N=n(), SE=2*sd(Value)/sqrt(N))

    return(summary)
}


make_figure_estimation_1 <- function(plot.K=4, plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                     save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation(41)
    }

    method.values.c = c("Standard-none-corrupted", "Adaptive (optimistic)-none-corrupted", "Adaptive (optimistic)-rho-epsilon-point-corrupted")
    method.labels.c = c("Standard", "Adaptive+", "Adaptive+ (plug-in)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(1,7,7)]
    shape.scale.c <- c(1,0,15)
    linetype.scale.c <- c(1,1,1)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(10000), epsilon_alpha==plot.epsilon_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")
    pp <- df %>%
        filter(Key %in% c("Coverage","Size")) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
#        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_%s_point_K%d.pdf", 10000, plot.guarantee, plot.K)
        ggsave(file=plot.file, height=3, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data_estimation(41)

make_figure_estimation_1(plot.K=2, plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)
make_figure_estimation_1(plot.K=4, plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)
make_figure_estimation_1(plot.K=8, plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)


make_figure_estimation_2 <- function(plot.K=4, plot.epsilon_max=0.2, plot.epsilon_train="corrupted", plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                     save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation(42)
    }

    method.values.c = c("Standard-none-corrupted", "Adaptive (optimistic)-none-corrupted", "Adaptive (optimistic)-rho-epsilon-point-corrupted",
                        sprintf("Adaptive (optimistic)-rho-epsilon-ci-pb-%s",plot.epsilon_train))
    method.labels.c = c("Standard", "Adaptive+", "Adaptive+ (plug-in)", "Adaptive+ (CI)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(1,7,7,8,9)]
    shape.scale.c <- c(1,0,15,8)
    linetype.scale.c <- c(1,1,1,1,1)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha,
               epsilon_max==plot.epsilon_max) %>%
        filter(n_cal %in% c(10000), epsilon_alpha==plot.epsilon_alpha) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        filter(Key %in% c("Coverage", "Size"))

    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")

    pp <- df %>%
        mutate(N_corr = sprintf("Corrupt samples: %d", epsilon_n_corr)) %>%
        mutate(N_corr = factor(N_corr, c("Corrupt samples: 1000", "Corrupt samples: 5000", "Corrupt samples: 10000"))) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        facet_grid(Key~N_corr, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
#        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="right")

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_%s_ci_pred_K%d_emax%s.pdf", 10000, plot.guarantee, plot.K, plot.epsilon_max)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_estimation_2_ci <- function(plot.K=4, plot.epsilon_max=0.2, plot.epsilon_train="corrupted", plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                        save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation(42)
    }

    method.values.c = c(sprintf("Adaptive (optimistic)-rho-epsilon-ci-pb-%s",plot.epsilon_train))
    method.labels.c = c("Adaptive+ (CI)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(7,8,9)]
    shape.scale.c <- c(25,24,17)
    linetype.scale.c <- c(1,2,3)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha,
               epsilon_max==plot.epsilon_max) %>%
        filter(n_cal %in% c(10000), epsilon_alpha==plot.epsilon_alpha) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        filter(Key %in% c("epsilon_low", "epsilon_upp")) %>%
        mutate(`Confidence bound` = factor(Key, c("epsilon_upp", "epsilon_low", "epsilon_hat"), c("Upper", "Lower", "Point estimate")))

#    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
#    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")

    pp <- df %>%
        mutate(N_corr = sprintf("Corrupt samples: %d", epsilon_n_corr)) %>%
        mutate(N_corr = factor(N_corr, c("Corrupt samples: 1000", "Corrupt samples: 5000", "Corrupt samples: 10000"))) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, shape=`Confidence bound`)) +
        geom_point() +
        geom_line(linetype=3) +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1, alpha=0.5) +
        facet_grid(Epsilon~N_corr, scales="free") +
        geom_hline(aes(yintercept=epsilon), color="red") +
#        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
#        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
#        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
#        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(legend.position="right")

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_%s_ci_bounds_K%d_emax%s.pdf", 10000, plot.guarantee, plot.K, plot.epsilon_max)
        ggsave(file=plot.file, height=2, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}


summary <- load_data_estimation(42)

plot.epsilon_train <- "corrupted"
plot.epsilon_max <- 0.25
plot.epsilon_alpha <- 0.01
plot.K <- 2
make_figure_estimation_2(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                         plot.epsilon_alpha=plot.epsilon_alpha, plot.guarantee="lab-cond",
                         save_plots=TRUE, reload=FALSE)
make_figure_estimation_2_ci(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                            plot.epsilon_alpha=plot.epsilon_alpha, plot.guarantee="lab-cond",
                            save_plots=TRUE, reload=FALSE)

plot.epsilon_max <- 0.2
make_figure_estimation_2(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                         plot.epsilon_alpha=plot.epsilon_alpha, plot.guarantee="lab-cond",
                         save_plots=TRUE, reload=FALSE)
make_figure_estimation_2_ci(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                            plot.epsilon_alpha=plot.epsilon_alpha, plot.guarantee="lab-cond",
                            save_plots=TRUE, reload=FALSE)



make_figure_estimation_2_slides <- function(plot.K=4, plot.epsilon_max=0.2, plot.epsilon_train="corrupted", plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                            version=1, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation(42)
    }

    method.values.c = c("Standard-none-corrupted", "Adaptive (optimistic)-none-corrupted", "Adaptive (optimistic)-rho-epsilon-point-corrupted",
                        sprintf("Adaptive (optimistic)-rho-epsilon-ci-pb-%s",plot.epsilon_train))
    method.labels.c = c("Standard", "Adaptive+", "Adaptive+ (plug-in)", "Adaptive+ (CI)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(1,7,7,8,9)]
    shape.scale.c <- c(1,0,15,8)
    linetype.scale.c <- c(1,1,1,1,1)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha,
               epsilon_max==plot.epsilon_max) %>%
        filter(n_cal %in% c(10000), epsilon_alpha==plot.epsilon_alpha) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        filter(Key %in% c("Coverage", "Size"))

    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage","Size","Size"), Mean=c(0.7,1,1.0,1.4), Method="Standard")

    if(version==1) {
        df <- df %>% filter(Method %in% c("Standard", "Adaptive+"))
    } else if(version==2) {
        df <- df %>% filter(Method %in% c("Standard", "Adaptive+", "Adaptive+ (plug-in)"))
    }
    
    pp <- df %>%
        mutate(N_corr = sprintf("Corrupt samples: %d", epsilon_n_corr)) %>%
        mutate(N_corr = factor(N_corr, c("Corrupt samples: 1000", "Corrupt samples: 5000", "Corrupt samples: 10000"))) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        facet_grid(Key~N_corr, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
#        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="bottom")

    if(save_plots) {
        plot.file <- sprintf("figures_slides/synthetic1_ntrain%d_%s_ci_pred_K%d_emax%s_%s.pdf", 10000, plot.guarantee, plot.K, plot.epsilon_max, version)
        ggsave(file=plot.file, height=4.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

plot.epsilon_train <- "corrupted"
plot.epsilon_max <- 0.25
plot.epsilon_alpha <- 0.01
plot.K <- 2
for(version in c(1,2,3)) {
    make_figure_estimation_2_slides(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                                    plot.epsilon_alpha=plot.epsilon_alpha, plot.guarantee="lab-cond",
                                    version=version, save_plots=TRUE, reload=FALSE)
}


make_figure_estimation_3 <- function(plot.contamination, plot.K=4, plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                     save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation(43)
    }

    method.values.c = c("Standard-none-corrupted", "Adaptive (optimistic)-none-corrupted", "Adaptive (optimistic)-rho-epsilon-point-corrupted")
    method.labels.c = c("Standard", "Adaptive+", "Adaptive+ (plug-in)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(1,7,7)]
    shape.scale.c <- c(1,0,15)
    linetype.scale.c <- c(1,1,1)


    df <- summary %>%
        filter(contamination==plot.contamination, data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(10000), epsilon_alpha==plot.epsilon_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")
    pp <- df %>%
        filter(Key %in% c("Coverage", "Size")) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
#        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ntrain%d_%s_point_K%d_%s.pdf", 10000, plot.guarantee, plot.K, plot.contamination)
        ggsave(file=plot.file, height=3, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data_estimation(43)

make_figure_estimation_3(plot.K=4, plot.contamination="block", plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)
make_figure_estimation_3(plot.K=4, plot.contamination="random", plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)


make_figure_estimation_3_slides <- function(plot.contamination, plot.K=4, plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                            version=1, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation(43)
    }

    method.values.c = c("Standard-none-corrupted", "Adaptive (optimistic)-none-corrupted", "Adaptive (optimistic)-rho-epsilon-point-corrupted")
    method.labels.c = c("Standard", "Adaptive+", "Adaptive+ (plug-in)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(1,7,7)]
    shape.scale.c <- c(1,0,15)
    linetype.scale.c <- c(1,1,1)


    df <- summary %>%
        filter(contamination==plot.contamination, data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal %in% c(10000), epsilon_alpha==plot.epsilon_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage", "Size", "Size"), Mean=c(0.7,1,1.4,2.1), Method="Standard")

    df <- df %>%
        filter(Key %in% c("Coverage", "Size")) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c))        

    if(version==1) {
        df <- df %>% filter(Method %in% c("Standard"))
    } else if(version==2) {
        df <- df %>% filter(Method %in% c("Standard", "Adaptive+"))
    }


    pp <- df %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        facet_grid(Key~Epsilon, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
#        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="bottom")

    if(save_plots) {
        plot.file <- sprintf("figures_slides/synthetic1_ntrain%d_%s_point_K%d_%s_%s.pdf", 10000, plot.guarantee, plot.K, plot.contamination, version)
        ggsave(file=plot.file, height=4.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}


for(version in c(1,2,3)) {
    make_figure_estimation_3_slides(plot.K=4, plot.contamination="block-const", plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                    version=version, save_plots=TRUE, reload=FALSE)
    make_figure_estimation_3_slides(plot.K=4, plot.contamination="random-const", plot.alpha=0.1, plot.epsilon_alpha=0.05, plot.guarantee="lab-cond",
                                    version=version, save_plots=TRUE, reload=FALSE)
}


######################################
## Bounded experiments (2-level RR) ##
######################################

# --------------------------------------------------
# Plot calibration-conditional coverage, as a function of calibration set size
# Bounded noise method


load_data_bounded <- function(exp.num) {
    idir <- sprintf("results_hpc/exp%d", exp.num)
    ifile.list <- list.files(idir)
    results <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(), guess_max=2)
    }))


    summary <- results %>%
        pivot_longer(c("Coverage", "Size"), names_to = "Key", values_to = "Value") %>%
        group_by(data, num_var, K, signal, model_name, contamination, epsilon, nu, estimate, n_train, n_cal, Guarantee, Alpha, Label, Method,
                 epsilon_max, epsilon_se, nu_max, nu_se, V_alpha, Key) %>%
        summarise(Mean=mean(Value), N=n(), SE=2*sd(Value)/sqrt(N))

    return(summary)
}

method.values = c("Standard", "Adaptive (optimistic)")
method.labels = c("Standard", "Adaptive+")
cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
df.dummy <- tibble(key="Coverage", value=0.95)
df.dummy2 <- tibble(key="Coverage", value=0.5)
color.scale <- cbPalette[c(1,7)]
shape.scale <- c(1,0)
linetype.scale <- c(1,1)

make_figure_bounded_1_brr <- function(plot.alpha=0.1, plot.epsilon=0.1, plot.guarantee="lab-cond", plot.n_cal=10000, plot.K=4,
                                      plot.epsilon_max=0.1, plot.nu_max=0.1, plot.nu_se=0,
                                      plot.V_alpha=0.01, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_bounded(310)
    }

    df <- summary %>%
        filter(Method != "Adaptive (pessimistic)") %>%
        filter(data=="s1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal == plot.n_cal,
               epsilon==plot.epsilon, epsilon_max==plot.epsilon_max,
               nu_se==plot.nu_se,
               nu_max==plot.nu_max,
               V_alpha==plot.V_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), Method="Standard")

    appender <- function(string) TeX(paste("$\\nu : $", string))  

    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("Contam: %.2f", epsilon)) %>%
        mutate(Nu = sprintf("%.2f", nu)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        mutate(epsilon_lower = epsilon - epsilon_se) %>%
        ggplot(aes(x=epsilon_lower, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Nu, scales="free", labeller = labeller(.default=label_parsed,Nu=appender)) +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(breaks=c(0.15,0.16,0.17,0.18,0.2)) +
#        scale_x_continuous(trans='log10') +
        xlab(TeX("Lower bound for the $\\epsilon$$~parameter")) +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ncal%d_K%d_%s_bounded_eps%.2f_emax%.2f_numax%.2f_ealpha%.2f_brr_eps_nuse%.2f.pdf",
                             plot.n_cal, plot.K, plot.guarantee, plot.epsilon, plot.epsilon_max, plot.nu_max,
                             plot.V_alpha, plot.nu_se)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data_bounded(310)
make_figure_bounded_1_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 10000, plot.nu_se=0,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_bounded_1_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 10000, plot.nu_se=0.02,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_bounded_1_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 100000, plot.nu_se=0,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_bounded_1_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 100000, plot.nu_se=0.02,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)


make_figure_bounded_2_brr <- function(plot.alpha=0.1, plot.epsilon=0.1, plot.guarantee="lab-cond", plot.n_cal=10000,
                                      plot.epsilon_se=0, plot.K=4,
                                      plot.epsilon_max=0.1, plot.nu_max=0.1, plot.V_alpha=0.01, save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_bounded(311)
    }

    df <- summary %>%
        filter(Method != "Adaptive (pessimistic)") %>%
        filter(data=="s1", num_var==50, n_train==10000, K==plot.K, signal==1, Guarantee==plot.guarantee,
               Label=="marginal", model_name=="RFC", Alpha==plot.alpha) %>%
        filter(n_cal == plot.n_cal,
               epsilon==plot.epsilon, epsilon_max==plot.epsilon_max, epsilon_se==plot.epsilon_se,
               nu_max==plot.nu_max,
               V_alpha==plot.V_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.8,1), Method="Standard")

    appender <- function(string) TeX(paste("$\\nu : $", string))  

    pp <- df %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Epsilon = sprintf("%.2f", epsilon)) %>%
        mutate(Nu = sprintf("%.2f", nu)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        mutate(nu_width = pmin(1, nu + nu_se) - pmax(0, nu - nu_se)) %>%
        ggplot(aes(x=nu_width, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Nu, scales="free", labeller = labeller(.default=label_parsed,Nu=appender)) +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
#        scale_x_continuous(breaks=c(0.15,0.16,0.17,0.18,0.2)) +
#        scale_x_continuous(trans='log10') +
        xlab(TeX("Width of confidence interval for the $\\nu$$~parameter")) +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ncal%d_K%d_%s_bounded_eps%.2f_emax%.2f_numax%.2f_ealpha%.2f_brr_nu_epsse%.2f.pdf",
                             plot.n_cal, plot.K, plot.guarantee, plot.epsilon, plot.epsilon_max, plot.nu_max,
                             plot.V_alpha, plot.epsilon_se)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data_bounded(311)
make_figure_bounded_2_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 10000, plot.epsilon_se=0,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_bounded_2_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 10000, plot.epsilon_se=0.02,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_bounded_2_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 100000, plot.epsilon_se=0,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)
make_figure_bounded_2_brr(plot.epsilon=0.2, plot.epsilon_max=0.2, plot.nu_max=1, plot.V_alpha=0.01,
                          plot.n_cal = 100000, plot.epsilon_se=0.02,
                          plot.alpha=0.1, plot.guarantee="lab-cond", plot.K=4, save_plots=TRUE, reload=FALSE)


#########################
## Estimation block-RR ##
#########################

load_data_estimation_BRR <- function(exp.num) {
    idir <- sprintf("results_hpc/exp%d", exp.num)
    ifile.list <- list.files(idir)
    results <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(), guess_max=2)
    }))


    summary <- results %>%
        pivot_longer(c("Coverage", "Size", "epsilon_low", "epsilon_upp", "epsilon_hat", "nu_low", "nu_upp", "nu_hat"), names_to = "Key", values_to = "Value") %>%
        group_by(data, num_var, K, signal, model_name, contamination, epsilon, nu, estimate, n_train, n_cal, Guarantee, Alpha, Label, Method,
                 epsilon_max, nu_max, V_alpha, epsilon_train, epsilon_n_clean, epsilon_n_corr, Key) %>%
        summarise(Mean=mean(Value), N=n(), SE=2*sd(Value)/sqrt(N))

    return(summary)
}


make_figure_estimation_1_BRR <- function(plot.K=4, plot.n_cal=10000, plot.alpha=0.1, plot.V_alpha=0.05, plot.guarantee="lab-cond",
                                         save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation(410)
    }

    method.values.c = c("Standard-none-corrupted", "Adaptive (optimistic)-none-corrupted", "Adaptive (optimistic)-rho-epsilon-point-corrupted")
    method.labels.c = c("Standard", "Adaptive+", "Adaptive+ (plug-in)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(1,7,7)]
    shape.scale.c <- c(1,0,15)
    linetype.scale.c <- c(1,1,1)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha, epsilon==plot.epsilon) %>%
        filter(n_cal %in% c(10000), V_alpha==plot.V_alpha)
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")

    appender <- function(string) TeX(string)
    
    pp <- df %>%
        filter(Key %in% c("Coverage","Size")) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Parameters = sprintf("($\\epsilon$: %.2f, $\\nu$: %.2f)", epsilon, nu)) %>%
#        mutate(Label = factor(Label, label.values, label.labels)) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE)) +
        facet_grid(Key~Parameters, scales="free", labeller = labeller(.default=label_parsed,Parameters=appender)) +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
#        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ncal%d_%s_point_K%d_BRR.pdf", plot.n_cal, plot.guarantee, plot.K)
        ggsave(file=plot.file, height=3, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data_estimation_BRR(410)

make_figure_estimation_1_BRR(plot.K=4, plot.alpha=0.1, plot.V_alpha=0.05, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)



make_figure_estimation_2_ci_BRR <- function(plot.K=4, plot.epsilon_max=0.2, plot.epsilon_train="corrupted",
                                            plot.alpha=0.1, plot.V_alpha=0.05, plot.guarantee="lab-cond",
                                            save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation_BRR(420)
    }

    method.values.c = c(sprintf("Adaptive (optimistic)-r-e-ci-pb-%s",plot.epsilon_train))
    method.labels.c = c("Adaptive+ (CI)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(7,8,9)]
    shape.scale.c <- c(25,24,17)
    linetype.scale.c <- c(1,2,3)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha,
               epsilon_max==plot.epsilon_max) %>%
        filter(n_cal == 10000, V_alpha==plot.V_alpha) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("$$\\epsilon$$: %.2f", epsilon)) %>%
        mutate(Nu = sprintf("$\\nu$$:~%.2f", nu)) %>%
        filter(Key %in% c("epsilon_low", "epsilon_upp")) %>%
        mutate(`Confidence bound` = factor(Key, c("epsilon_upp", "epsilon_low", "epsilon_hat"), c("Upper", "Lower", "Point estimate")))

#    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
#    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")
    appender <- function(string) TeX(string)

    pp <- df %>%
        mutate(N_corr = sprintf("Corrupt samples: %d", epsilon_n_corr)) %>%
        mutate(N_corr = factor(N_corr, c("Corrupt samples: 1000", "Corrupt samples: 5000", "Corrupt samples: 10000"))) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, shape=`Confidence bound`)) +
        geom_point() +
        geom_line(linetype=3) +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1, alpha=0.5) +
        facet_grid(Epsilon~Nu, scales="fixed", labeller = labeller(.default=label_parsed,Epsilon=appender,Nu=appender)) +
        geom_hline(aes(yintercept=epsilon), color="red") +
#        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
#        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
#        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
        scale_y_continuous(limits=c(0,0.3)) +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_%s_ci_bounds_K%d_emax%s_BRR.pdf", plot.guarantee, plot.K, plot.epsilon_max)
        ggsave(file=plot.file, height=3, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_estimation_2_nu_ci_BRR <- function(plot.K=4, plot.epsilon_max=0.2, plot.epsilon_train="corrupted",
                                               plot.alpha=0.1, plot.V_alpha=0.05, plot.guarantee="lab-cond",
                                               save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation_BRR(420)
    }

    method.values.c = c(sprintf("Adaptive (optimistic)-r-e-ci-pb-%s",plot.epsilon_train))
    method.labels.c = c("Adaptive+ (CI)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(7,8,9)]
    shape.scale.c <- c(25,24,17)
    linetype.scale.c <- c(1,2,3)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha,
               epsilon_max==plot.epsilon_max) %>%
        filter(n_cal == 10000, V_alpha==plot.V_alpha) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("$$\\epsilon$$: %.2f", epsilon)) %>%
        mutate(Nu = sprintf("$\\nu$$:~%.2f", nu)) %>%
        filter(Key %in% c("nu_low", "nu_upp")) %>%
        mutate(`Confidence bound` = factor(Key, c("nu_upp", "nu_low", "nu_hat"), c("Upper", "Lower", "Point estimate")))

#    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
#    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")
    appender <- function(string) TeX(string)

    pp <- df %>%
        mutate(N_corr = sprintf("Corrupt samples: %d", epsilon_n_corr)) %>%
        mutate(N_corr = factor(N_corr, c("Corrupt samples: 1000", "Corrupt samples: 5000", "Corrupt samples: 10000"))) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, shape=`Confidence bound`)) +
        geom_point() +
        geom_line(linetype=3) +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1, alpha=0.5) +
        facet_grid(Epsilon~Nu, scales="fixed", labeller = labeller(.default=label_parsed,Epsilon=appender,Nu=appender)) +
        geom_hline(aes(yintercept=nu), color="red") +
#        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
#        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
#        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +    
        scale_y_continuous(limits=c(0,1)) +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_%s_ci_nu_bounds_K%d_emax%s_BRR.pdf", plot.guarantee, plot.K, plot.epsilon_max)
        ggsave(file=plot.file, height=3, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}


make_figure_estimation_2_BRR <- function(plot.n_cal=10000, plot.K=4, plot.epsilon_max=0.2, plot.epsilon_train="corrupted", plot.alpha=0.1, plot.V_alpha=0.05,
                                         plot.epsilon=0.1, plot.guarantee="lab-cond",
                                         save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data_estimation_BRR(420)
    }

    method.values.c = c("Standard-none-corrupted", "Adaptive (optimistic)-none-corrupted", "Adaptive (optimistic)-r-e-p-corrupted",
                        sprintf("Adaptive (optimistic)-r-e-ci-pb-%s",plot.epsilon_train))
    method.labels.c = c("Standard", "Adaptive+", "Adaptive+ (plug-in)", "Adaptive+ (CI)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    color.scale.c <- cbPalette[c(1,7,7,8,9)]
    shape.scale.c <- c(1,0,15,8)
    linetype.scale.c <- c(1,1,1,1,1)


    df <- summary %>%
        filter(data=="s1", num_var==50, n_train==10000, signal==1, Guarantee==plot.guarantee,
               K==plot.K,Label=="marginal", model_name=="RFC", Alpha==plot.alpha,
               epsilon_max==plot.epsilon_max, epsilon==plot.epsilon) %>%
        filter(n_cal == plot.n_cal, V_alpha==plot.V_alpha) %>%
        filter(epsilon_n_clean >= 100) %>%
        mutate(Method = sprintf("%s-%s-%s", Method, estimate, epsilon_train)) %>%
        filter(Method %in% method.values.c) %>%
        mutate(Method = factor(Method, method.values.c, method.labels.c)) %>%
        mutate(Epsilon = sprintf("$$\\epsilon$$: %.2f", epsilon)) %>%
        mutate(Nu = sprintf("$\\nu$$:~%.2f", nu)) %>%
        filter(Key %in% c("Coverage", "Size"))

    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.7,1), Method="Standard")

    appender <- function(string) TeX(string)

    pp <- df %>%
        mutate(N_corr = sprintf("Corrupt samples: %d", epsilon_n_corr)) %>%
        mutate(N_corr = factor(N_corr, c("Corrupt samples: 1000", "Corrupt samples: 5000", "Corrupt samples: 10000"))) %>%
        ggplot(aes(x=epsilon_n_clean, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        facet_grid(Key~Nu, scales="free", labeller = labeller(.default=label_parsed,Epsilon=appender,Nu=appender)) +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(y=Mean), x=0, alpha=0) +
        scale_color_manual(values=color.scale.c) +
        scale_shape_manual(values=shape.scale.c) +
        scale_linetype_manual(values=linetype.scale.c) +
        scale_x_continuous(trans='log10') +
                                        #        scale_x_continuous(trans='log10') +
        xlab("Number of clean samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="right")

    if(save_plots) {
        plot.file <- sprintf("figures/synthetic1_ncal%d_%s_ci_pred_K%d_emax%s_eps%.2f_BRR.pdf", plot.n_cal, plot.guarantee, plot.K, plot.epsilon_max, plot.epsilon)
        ggsave(file=plot.file, height=3.5, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data_estimation_BRR(420)

plot.epsilon_train <- "corrupted"
plot.epsilon_max <- 0.25
plot.V_alpha <- 0.01
plot.K <- 4
make_figure_estimation_2_nu_ci_BRR(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                                   plot.V_alpha=plot.V_alpha, plot.guarantee="lab-cond",
                                   save_plots=TRUE, reload=FALSE)


if(FALSE) {
    make_figure_estimation_2_BRR(plot.n_cal=100000, plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                                 plot.V_alpha=plot.V_alpha, plot.guarantee="lab-cond", plot.epsilon=0.2,
                                 save_plots=TRUE, reload=FALSE)
}



for(plot.K in c(4,8,16)) {
    make_figure_estimation_2_BRR(plot.n_cal=10000, plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                                 plot.V_alpha=plot.V_alpha, plot.guarantee="lab-cond", plot.epsilon=0.2,
                                 save_plots=TRUE, reload=FALSE)

        make_figure_estimation_2_BRR(plot.n_cal=10000, plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                                 plot.V_alpha=plot.V_alpha, plot.guarantee="lab-cond", plot.epsilon=0.1,
                                 save_plots=TRUE, reload=FALSE)

    make_figure_estimation_2_ci_BRR(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                                    plot.V_alpha=plot.V_alpha, plot.guarantee="lab-cond",
                                    save_plots=TRUE, reload=FALSE)

    make_figure_estimation_2_nu_ci_BRR(plot.K=plot.K, plot.epsilon_max=plot.epsilon_max, plot.epsilon_train=plot.epsilon_train, plot.alpha=0.1,
                                       plot.V_alpha=plot.V_alpha, plot.guarantee="lab-cond",
                                       save_plots=TRUE, reload=FALSE)

}
