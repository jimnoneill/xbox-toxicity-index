# Observed annual-target messages published before the post-runoff cutoff.
base <- "/home/joneill/x_box/pipeline/data/kistner"
out <- "/home/joneill/x_box/pipeline/manuscript/epjds_submission_work/external"
e <- new.env()
load(file.path(base, "raw/Scaled.Rda"), e)
d <- e$comms_df
rows <- list()
for (name in c("tw", "twfb")) {
    t <- read.csv(file.path(base, paste0("transitions_", name, ".csv")))
    ids <- unique(t$MemberICPSR[t$Y == 2021])
    modes <- if (name == "tw") "Twitter" else c("Twitter", "Facebook")
    selected <- d$Year == 2021 & d$Mode %in% modes & d$MemberICPSR %in% ids
    early <- selected & d$Date < as.Date("2021-01-20")
    rows[[name]] <- data.frame(measure=name, total_messages=sum(selected),
        early_messages=sum(early), early_members=length(unique(d$MemberICPSR[early])),
        scored_members=length(ids))
}
write.csv(do.call(rbind, rows), file.path(out, "jan_window_2021.csv"), row.names=FALSE)
print(do.call(rbind, rows))
