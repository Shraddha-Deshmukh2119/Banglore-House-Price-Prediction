function getBathValue() {
  var uiBathrooms = document.getElementsByName("uiBathrooms");
  for (var i = 0; i < uiBathrooms.length; i++) {
      if (uiBathrooms[i].checked) {
          return parseInt(uiBathrooms[i].value); // Get actual value instead of index
      }
  }
  return -1; // Invalid Value
}

function getBHKValue() {
  var uiBHK = document.getElementsByName("uiBHK");
  for (var i = 0; i < uiBHK.length; i++) {
      if (uiBHK[i].checked) {
          return parseInt(uiBHK[i].value); // Get actual value instead of index
      }
  }
  return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");

  var sqft = document.getElementById("uiSqft");
  var bhk = getBHKValue();
  var bathrooms = getBathValue();
  var location = document.getElementById("uiLocations");
  var estPrice = document.getElementById("uiEstimatedPrice");

  if (bhk === -1 || bathrooms === -1 || sqft.value.trim() === "" || location.value === "") {
      alert("Please enter valid inputs");
      return;
  }

  var url = "http://127.0.0.1:5000/predict_home_price"; // Change this if using Nginx

  $.post(url, {
      total_sqft: parseFloat(sqft.value),
      bhk: bhk,
      bath: bathrooms,
      location: location.value
  }, function (data, status) {
      console.log(data);
      if (data.estimated_price) {
          estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
      } else {
          estPrice.innerHTML = "<h2>Error predicting price</h2>";
      }
      console.log(status);
  }).fail(function () {
      estPrice.innerHTML = "<h2>Error connecting to server</h2>";
  });
}

function onPageLoad() {
  console.log("document loaded");
  var url = "http://127.0.0.1:5000/get_location_names"; // Change this if using Nginx

  $.get(url, function (data, status) {
      console.log("got response for get_location_names request");
      if (data && data.locations) {
          var uiLocations = document.getElementById("uiLocations");
          $('#uiLocations').empty();
          for (var i in data.locations) {
              var opt = new Option(data.locations[i]);
              $('#uiLocations').append(opt);
          }
      }
  });
}

window.onload = onPageLoad;
