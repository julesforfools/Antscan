##### 0. Notes #####
# This R-Script reads in the Antscan metadata Excel Spreadsheet and updates it according to Antcat.
# It further adds multiple summary columns that were added the first time this code was run.
# The entire script can be seen as a function that reads in the metadata and returns an updated version of itself.
# If there are no updates, the final check should state that the metadata were up-to-date.

##### 1. Setup #####
options(timeout = 1000) # for slow internet
# Load routines and functions
pkgs <- c("rstudioapi", "tidyverse", "readxl", "ape", "rgl", "viridis", "svglite", "httr", "RCurl", "RJSONIO", "xlsx", "readr")
sapply(pkgs, library, character.only = TRUE)

#Download metadata: https://docs.google.com/spreadsheets/d/1p28GF8kvA57a6Ao2aDKufdGiF5pd-sS3Y1FlSlvGXcU/edit?gid=1992961542
antscan <- read_excel(path = "C:/Users/katsu/OneDrive/University_Stuff/phd/data/antscan/antscan_metadata.xlsx", sheet = "antscan_metadata") # Excel files must be closed

legend <- read_excel(path = "C:/Users/katsu/OneDrive/University_Stuff/phd/data/antscan/antscan_metadata.xlsx", sheet = "legend") # Excel files must be closed
antscan_old <- antscan # Create a copy to compare changes at the end

# create function that checks taxa on Antcat
check_taxon <- function(genus, species){
  # Check for unidentified specimens
  if (grepl("sp\\.", species) | # If there is "sp." in the species name
      grepl("cf\\.", species) | # If there is "cf." in the species name
      grepl("aff\\.", species) | # If there is "aff." in the species name
      grepl("indet", species) | # If there is "indet." in the species name
      grepl("[0-9]", species) # If there is a number in the species name
  ){
    return(c("unidentified" ,paste(genus, species, sep = " "))) # We assume the species is unidentified
  } else {
    # Clean subspecies name, replace "." with "%20"
    species <- gsub("\\.", " ", x = species)
    file_url <- paste("https://antcat.org/v1/taxa/search/", genus, "%20", species, sep = "")
    file_url <- gsub(" ", "%20", x = file_url)
    #httr, RJSONIO
    #results <- GET(file_url)
    #content <- (content(results, "text"))
    #content <- fromJSON(content)
    #Rcurl, RJSONIO #Rcurl has problems on windows with ssh, set verifypeer to FALSE!
    results <- getURL(file_url, ssl.verifypeer = FALSE)
    # If accessing the URL fails or no results, exit early
    if(results == "[]" | results == "" | grepl("HTTP 404", results)) {
      print(paste(genus, species, " is not an ant, unidentified, too much info, too new, or a typo in the name", sep = " "))
      return(c("invalid" ,paste(genus, species, sep = " ")))
    } 
    content <- fromJSON(results)
    # If one result, the case should be clear
    if(length(content) == 1) { 
      ant_ID <- as.character(content[[1]]$id)
      file_url_2 <- paste("https://antcat.org/v1/taxa/", ant_ID, sep = "")
      results_2 <- getURL(file_url_2, ssl.verifypeer = FALSE)
      content_2 <- fromJSON(results_2)
      # Check Validity of current taxon
      if (content_2[[1]]$status == "valid") {
        return(c(content_2[[1]]$status, content_2[[1]]$name_cache))
      } else {
        print(paste(genus, species, "is not a valid taxon", sep = " "))
        ant_ID_valid <- as.character(content_2[[1]]$current_taxon_id)
        file_url_3 <- paste("https://antcat.org/v1/taxa/", ant_ID_valid, sep = "")
        results_3 <- getURL(file_url_3, ssl.verifypeer = FALSE)
        content_3 <- fromJSON(results_3)
        return(c(content_2[[1]]$status, content_3[[1]]$name_cache))
      }
      # If there's more results, then there is a synonym or subspecies or something else involved  
    } else { 
      ant_ID_multi <- c()
      for (i in content) {ant_ID_multi <- c(ant_ID_multi, (i$id))}
      ant_URL_multi <- c()
      for (i in ant_ID_multi) {ant_URL_multi<-c(ant_URL_multi,paste("https://antcat.org/v1/taxa/", i, sep = ""))}
      ant_cat_multi <- lapply(ant_URL_multi, getURL, ssl.verifypeer = F)
      ant_cont_multi <- lapply(ant_cat_multi, fromJSON)
      ant_cont_match <- list()
      for (i in ant_cont_multi) {
        # Has to pass quality check: exact name match and no synonyms allowed!
        if (paste(genus, species) == i[[1]]$name_cache & i[[1]]$status != "synonym") { 
          ant_cont_match <- c(ant_cont_match, i)}
      }
      if (length(ant_cont_match) == 0) {
        print(paste(genus, species, " is not an ant, unidentified, too much info, too new, or a typo in the name", sep = " "))
        return(c("invalid" ,paste(genus, species, sep = " ")))
      } 
      if (length(ant_cont_match) > 1) {
        print(paste(genus, species, "Watch out this! This species might be a taxonomic mess!"))}
      ant_cont_match <- ant_cont_match[[1]] # Make it lazy at this point.
      if (ant_cont_match$status == "valid") {
        return(c(ant_cont_match$status, ant_cont_match$name_cache))
      } else {
        print(paste(genus, species, "is not a valid taxon", sep = " "))
        ant_ID_valid <- as.character(ant_cont_match$current_taxon_id)
        file_url_3 <- paste("https://antcat.org/v1/taxa/", ant_ID_valid, sep = "")
        results_3 <- getURL(file_url_3, ssl.verifypeer = FALSE)
        content_3 <- fromJSON(results_3)
        return(c(ant_cont_match$status, content_3[[1]]$name_cache))
      }
    }
  }
}


##### 2. Clean data #####

### Pre - Cleaning steps ###
antscan$taxon_code <- gsub(x=antscan$taxon_code, pattern = "Pyramica\\.", replacement = "Strumigenys\\.") # rename Pyramica
antscan$taxon_code <- gsub(x=antscan$taxon_code, pattern = "Dorylus\\.\\(Typhlopone\\)\\.fulvus", replacement = "Dorylus\\.fulvus") # rename Dorylus.(Typhlopone).fulvus
#### create Genus & Species, clean Taxon code ####
antscan$taxon_code <- gsub(x=antscan$taxon_code, pattern = " ", replacement = "\\.") # clean taxon code
antscan$taxon_code <- gsub(x=antscan$taxon_code, pattern = "(sp$)", replacement = "sp\\.") # clean taxon code
antscan$taxon_code <- gsub(x=antscan$taxon_code, pattern = "\\.\\.", replacement = "\\.") # clean taxon code
antscan$Genus <- gsub(x=antscan$taxon_code, pattern = "(.*?)\\..*", replacement = "\\1") # create Genus
antscan$Species <- gsub(x=antscan$taxon_code, pattern = ".*?\\.(.*)", replacement = "\\1") # create Species


#### 3. create Caste ####
#### create Caste
unique(antscan$lifestagesex_original_col)
antscan$caste <- antscan$lifestagesex_original_col
antscan$caste <- case_when(
  antscan$LTS_Box == "LTS OIST 04" ~ "outgroup",
  antscan$lifestagesex_original_col == "w" ~ "worker",
  antscan$lifestagesex_original_col == "Q" ~ "queen",
  antscan$lifestagesex_original_col == "aQ" ~ "queen",
  antscan$lifestagesex_original_col == "m" ~ "male",
  antscan$lifestagesex_original_col == "Gynandromorph" ~ "gynandromorph",
  antscan$lifestagesex_original_col == "Intercaste" ~ "worker",
  antscan$lifestagesex_original_col == "In copula" ~ "queen",
  antscan$lifestagesex_original_col == "Minor w" ~ "worker",
  antscan$lifestagesex_original_col == "Ergatoid m" ~ "male",
  antscan$lifestagesex_original_col == "alate" ~ "undetermined",
  antscan$lifestagesex_original_col == "Q ergatoid" ~ "queen",
  antscan$lifestagesex_original_col == "W" ~ "worker",
  antscan$lifestagesex_original_col == "larva" ~ "larva",
  antscan$lifestagesex_original_col == "major" ~ "worker",
  antscan$lifestagesex_original_col == "pupa" ~ "pupa",
  antscan$lifestagesex_original_col == "ergatoid Q" ~ "queen",
  antscan$lifestagesex_original_col == "w?" ~ "worker",
  antscan$lifestagesex_original_col == "w, brood between mandible" ~ "worker",
  antscan$lifestagesex_original_col == "w (broken)" ~ "worker",
  antscan$lifestagesex_original_col == "w, with brood between mandible" ~ "worker",
  antscan$lifestagesex_original_col == "Q?" ~ "queen",
  antscan$lifestagesex_original_col == "m?" ~ "male",
  antscan$lifestagesex_original_col == "m? aQ?" ~ "undetermined",
  antscan$lifestagesex_original_col == "maj w? Q?" ~ "undetermined",
  antscan$lifestagesex_original_col == "f" ~ "undetermined",
  is.na(antscan$lifestagesex_original_col) ~ "undetermined",
)
unique(antscan$caste)

#### create caste_detail
antscan$caste_detail <- antscan$subcaste_original_col
antscan$caste_detail <- case_when(
  antscan$LTS_Box == "LTS OIST 04" ~ "outgroup",
  antscan$caste_detail == "minor" ~ "minor",
  antscan$caste_detail == "min" ~ "minor",
  antscan$caste_detail == "media" ~ "media",
  antscan$caste_detail == "med" ~ "media",
  antscan$caste_detail == "intermediate" ~ "media",
  antscan$caste_detail == "inter" ~ "media",
  antscan$caste_detail == "major" ~ "major",
  antscan$caste_detail == "maj" ~ "major",
  antscan$caste_detail == "soldier" ~ "major",
  antscan$caste_detail == "larva" ~ "not applicable",
  antscan$caste_detail == "pupa" ~ "not applicable",
  antscan$lifestagesex_original_col == "aQ" ~ "alate",
  antscan$lifestagesex_original_col == "alate" ~ "alate",
  grepl(pattern = "ergatoid", x = antscan$lifestagesex_original_col) ~ "ergatoid",
  grepl(pattern = "Ergatoid", x = antscan$lifestagesex_original_col) ~ "ergatoid",
  grepl(pattern = "Intercaste", x = antscan$lifestagesex_original_col) ~ "intercaste",
  grepl(pattern = "intercaste", x = antscan$lifestagesex_original_col) ~ "intercaste",
  antscan$lifestagesex_original_col == "major" ~ "major",
  antscan$lifestagesex_original_col == "Minor w" ~ "minor",
  antscan$lifestagesex_original_col == "In copula" ~ "in_copula",
  #is.na(antscan$caste_detail) ~ "undetermined",
)
unique(antscan$caste_detail)

#### create Family_or_other
antscan$Family_or_other <- antscan$caste
antscan$Family_or_other <- case_when(
  antscan$caste != "outgroup" ~ "Formicidae",
  antscan$caste == "outgroup" ~ antscan$subcaste_original_col
)

#### Clean up colnames for transparency and more informativeness
antscan <- antscan %>% 
  relocate(caste, .after = subcaste) %>% 
  relocate(caste_detail, .after = caste)

##### 4. Update Subfamily and other info from antwiki #####
antwiki <- read_delim("https://www.antwiki.org/wiki/images/a/ad/AntWiki_Valid_Genera.txt", delim = "\t", locale = locale(encoding = "UTF-16LE"))

# drop unnecessary rows
antwiki <- antwiki %>%
  filter(is.na(Subgenus))
# Update antscan with antwiki columns
colnames(antwiki)[which(colnames(antwiki) == "GenusAuthority")] <- "Genus_authority"
# Remove unnecessary columns
drop_cols_2 <- c("TaxonName", "Subgenus", "Author", "Year", "OriginalTypeSpecies", "CurrentTypeSpecies", "SpeciesCount", "Images")
antwiki <- antwiki %>% select(-one_of(drop_cols_2))

# Update
antscan <- rows_update(antscan, antwiki, by = "Genus", unmatched = "ignore")


##### 5. Check taxon status #####
# Test Runs
check_taxon("Camponotus", "gigas")
check_taxon("Atta", "robusta")
check_taxon("Pachycondyla", "villosa")
check_taxon("Patagonomyrmex", "angustus")
check_taxon("Apis", "mellifera")
check_taxon("????", "sp.")
check_taxon("Anochetus", "gr.inermis.sp.n")
check_taxon("Camponotus", "rufoglaucus.zanzibaricus")
check_taxon("Dorylus", "(Typhlopone).fulvus")

antscan_antcat_lst <- mapply(check_taxon, antscan$Genus, antscan$Species)
antscan_antcat_df <- data.frame(matrix(unlist(antscan_antcat_lst), nrow=length(antscan_antcat_lst)/2, ncol = 2, byrow=TRUE))
colnames(antscan_antcat_df) <- c("status", "name_recommended")

antscan <- antscan %>% 
  mutate(Status = antscan_antcat_df$status) %>% 
  relocate(Status, .after = Species) %>% 
  mutate(Name = antscan_antcat_df$name_recommended) %>% 
  relocate(Name, .after = Status)


##### 6. Export #####
combined <- list(antscan_metadata = antscan, legend = legend)
writexl::write_xlsx(combined, path = paste0("antscan_metadata_", format(Sys.time(), "%y-%m-%d-%H%M%S"), ".xlsx"))
