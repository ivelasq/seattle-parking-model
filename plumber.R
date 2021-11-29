library(plumber)
library(pins)
library(tibble)
library(xgboost)
library(lubridate)
library(dplyr)
library(tidyr)
library(tidymodels)
library(recipes)
library(plumbertableau)
library(memoise)

rsc <-
  pins::board_rsconnect(server = Sys.getenv("CONNECT_SERVER"),
                        key = Sys.getenv("CONNECT_API_KEY"))

xgboost_model <-
  pins::pin_read("isabella.velasquez/seattle_parking_model", board = rsc)

pred_model <-
  function(block_id, ndays) {
  
  times <- Sys.time() + lubridate::ddays(ndays)
  
  current_time <-
    tibble::tibble(times = times,
                   id = block_id)
  
  current_prediction  <-
    current_time %>%
    transmute(
      id = id,
      hour = hour(times),
      month = month(times),
      dow = wday(times),
      occupancy_pct = NA
    ) %>%
    bake(xgboost_model$recipe, .)
  
  parking_prediction <- 
    xgboost_model$prediction_fit %>% 
    predict(new_data = current_prediction)
  
  predictions <-
    parking_prediction$.pred
  
  predictions[[1]]
  
  } 

pred_mem <- memoise(pred_model)

#* @apiTitle Seattle Parking Occupancy Percentage Prediction API
#* @apiDescription Return the predicted occupancy percentage at various Seattle locations

#* @tableauArg block_id:integer numeric block ID
#* @tableauArg ndays:integer number of days in the future for the prediction

#* @tableauReturn [numeric] Predicted occupancy rate
#* @post /pred

function(block_id, ndays) {
  pred_mem(as.numeric(block_id), as.numeric(ndays))
}

#* @plumber
tableau_extension
