console.log("Using Tensorflow Version: " + tf.version.tfjs + ", and Tensorflow Visor");

var dataMain;
var cleanDataMain = [];
var cleanDataTarget = [];

var data1;
var data2;

var makeValue;
var yearValue;
var odometerValue;
var bodyTypeValue;
var fuelValue;
var fuelConsumptionValue;
var driveTrainValue;
var transmissionValue;
var seatsValue;
var colourValue;

function retrieveFeatureValues(){
    makeValue = +document.getElementById("Make").value;
    yearValue = +document.getElementById("Year").value;
    odometerValue = +document.getElementById("Odometer").value;
    bodyTypeValue = +document.getElementById("BodyType").value;
    fuelValue = +document.getElementById("Fuel").value;
    fuelConsumptionValue = +document.getElementById("FuelConsumption").value;
    driveTrainValue = +document.getElementById("DriveTrain").value;
    transmissionValue = +document.getElementById("Transmission").value;
    seatsValue = +document.getElementById("Seats").value;
    colourValue = +document.getElementById("Colour").value;

    return tf.tensor([[makeValue, yearValue, odometerValue, bodyTypeValue, fuelValue, fuelConsumptionValue, driveTrainValue, transmissionValue, seatsValue, colourValue]])
}

const ALPHA = 0.001
const HIDDEN_SIZE = 4
const model = tf.sequential()

const myForm = document.getElementById("myForm");
const csvFile = document.getElementById("csvFile");

myForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const input = csvFile.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        const text = e.target.result;
        const data = d3.csvParse(text);
        dataMain = data;
        
        //Car Prices
        for(i = 1; i < dataMain.length; i++){
            cleanDataMain.push([+dataMain[i].Odometer, +dataMain[i].Make, +dataMain[i].Year, +dataMain[i].BodyType, +dataMain[i].Drivetrain, +dataMain[i].Fuel, +dataMain[i].Transmission, +dataMain[i].Colour, +dataMain[i].Seats, +dataMain[i].FuelConsumption]);
            cleanDataTarget.push(+dataMain[i].Price);
        }
        data1 = tf.tensor(cleanDataMain);
        data2 = tf.expandDims(tf.tensor(cleanDataTarget), 1);

        model.add(
            tf.layers.dense({
                inputShape: [data1.shape[1]],
                units: HIDDEN_SIZE,
                activation: "tanh",
            })
            )
            
            model.add(
                tf.layers.dense({
                    units: HIDDEN_SIZE,
                    activation: "tanh",
                })
            )
            
            model.add(
                tf.layers.dense({
                    units: 1,
                })
            )
            
            const train = async () => {
                console.log("Started Training...")
                model.compile({ optimizer: tf.train.sgd(ALPHA), loss: "meanSquaredError" })
                await model.fit(data1, data2, {
                    epochs: 100,
                    callbacks: {
                        onEpochEnd: async (epoch, logs) => {
                        if ((epoch + 1) % 10 === 0) {
                            console.log(`Epoch ${epoch + 1}: error: ${logs.loss}`)
                        }
                        },
                    },
                })
            }
            
            if (document.readyState !== "loading"){
                train();
            }else {
                document.addEventListener("DOMContentLoaded", train)
            }
    };
    reader.readAsText(input);
});

function predictFeatures(){
    let v = retrieveFeatureValues();
    console.log("Retrieved Features:");
    console.log(v.data());
    document.getElementById("outPutArea").innerHTML = model.predict(v).toString();
    console.log(model.predict(v).toString());
}