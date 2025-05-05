
# This script processes World Values Survey data for Indonesia to create prompts for LLM prediction.
# The WVS data file can be downloaded from: https://www.worldvaluessurvey.org/WVSEVStrend.jsp


# STEP 1: Load the file
file_path <- file.choose()  # Opens a file dialog
load(file_path)
wvs <- `WVS_Time_Series_1981-2022_v5_0`

# STEP 2: Select variables (respondent ID, country code, year, gender, age, income, religion)
df <- wvs[, c("S007", "S003", "S020", "X001", "X003", "X047R_WVS", "F025_WVS",  # IDs and demographics
              "A004", "A006", "E290", "F114E", "E150", "E114", "E069_01", "E069_11")]  # questions

# STEP 3: Filter only Indonesia (ISO code 360)
df <- df[df$S003 == 360, ]

# STEP 4: Handle missing values (negative values to NA)
df[df < 0] <- NA

# STEP 5: Simplify variable names
names(df) <- c("respondent_id", "country_code", "year", "gender", "age", "income", "religion",
               "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8")

# STEP 6: Map question texts
questions <- data.frame(
  question_code = paste0("q", 1:8),
  question_text = c(
    "How important is politics in your life?",
    "How important is religion in your life?",
    "Is political violence ever justifiable?",
    "Is terrorism as a political or ideological means justifiable?",
    "How often do you follow politics in the news on television or on the radio or in the daily papers?",
    "Is it good to have a strong leader who does not have to bother with parliament and elections?",
    "How much confidence do you have in religious institutions (Mosque, temple, church)?",
    "How much confidence do you have in the government?"
  )
)

# STEP 7: Convert to long format
library(dplyr)
library(tidyr)
df$id <- 1:nrow(df)
long_df <- df %>%
  pivot_longer(cols = starts_with("q"), names_to = "question_code", values_to = "answer") %>%
  filter(!is.na(answer)) %>%
  left_join(questions, by = "question_code")

# STEP 8: Label gender/income
long_df$gender_label <- ifelse(long_df$gender == 1, "male",
                               ifelse(long_df$gender == 2, "female", "unknown"))
long_df$income_label <- ifelse(long_df$income == 1, "low",
                               ifelse(long_df$income == 2, "middle",
                                      ifelse(long_df$income == 3, "high", "unknown")))

# STEP 9: Generate prompts
long_df$prompt <- paste0(
  "A ", long_df$age, "-year-old ", long_df$gender_label,
  " with ", long_df$income_label, " income, affiliated with religion code ", long_df$religion,
  " in Indonesia in ", long_df$year,
  " was asked: ", long_df$question_text
)

# STEP 10: Extract only needed columns and save
final_df <- long_df %>% select(prompt, answer)
#write.csv(final_df, "~/Downloads/wvs_prompt_response_indonesia.csv", row.names = FALSE)
nrow(final_df)

# Country trying to predict: Indonesia/ wvs$COUNTRY_ALP: IDN, wvs$S003: 360, wvs$COW_NUM: 850
#1-1. A004.- Important in life: Politics
summary(wvs$A004)
#1-2. A006.- Important in life: Religion 
summary(wvs$A006)
#2-1. E290.- Justifiable: Political violence
summary(wvs$E290)
#2-2. F114E.- Justifiable: Terrorism as a political, ideological or religious mean
summary(wvs$F114E)
#3. E150.- How often follows politics in the news
summary(wvs$E150)
#4. E114.- Political system: Having a strong leader
summary(wvs$E114)
#5-1. E069_01.- Confidence: Churches (Mosque, temple, church)
summary(wvs$E069_01)
#5-2. E069_11.- Confidence: The Government
summary(wvs$E069_11)
# Demographics-Country
summary(wvs$S003)
# Demographics-Year survey
summary(wvs$S020)
# Demographics-Unified respondent number
summary(wvs$S007)
# Demographics-Gender
summary(wvs$X001)
# Demographics-Age
summary(wvs$X003)
# Demographics-Region
summary(wvs$F025_WVS)
# Demographics-Income 
summary(wvs$X047R_WVS)