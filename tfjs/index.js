



// This tiny example illustrates how little code is necessary build /
// train / predict from a model in TensorFlow.js.  Edit this code
// and refresh the index.html to quickly explore the API.

// Tiny TFJS train / predict example.
async function run() {
  // Create a simple model.
// Use `tfjs`.

  const model = await tf.loadLayersModel('f/model.json');
 // const model = await tf.loadGraphModel('model.json');


  const img = new Image ();
  img.src ='beer3.jpg' 
  img.crossOrigin = 'anonymous';

  img.onload = async () => {

    let tensor = tf.browser.fromPixels(img, 3);
    // const float32Tensor = tensor.cast('float32');

    console.log(tensor.shape)
    // tf.expand_dims(img, axis=1) 

    console.log(tensor);
    tensor.print();




  

    console.log("Waited 5s");


    ///tensor = tf.cast(tensor, tf.float32);
    //console.log("Tensor dtype after casting: ", tensor.dtype);
   


        // //convert the image data to a tensor 
        // let tensor = tf.fromPixels(img)
        // //resize to 50 X 50
        // const resized = tf.image.resizeBilinear(tensor, [50, 50]).toFloat()
        // // Normalize the image 
        // const offset = tf.scalar(255.0);
        // const normalized = tf.scalar(1.0).sub(resized.div(offset));
        // //We add a dimension to get a batch shape 
        // const batched = normalized.expandDims(0)
        // return batched

    // tensor = tf.image.resizeBilinear(tensor, [224, 224])

    tensor = tf.image.resizeBilinear(tensor, [224, 224]).div(tf.scalar(255))
    tensor = tf.cast(tensor, dtype = 'float32');

    // tensor = tensor / 255.0;
    // console.log(tensor);

    // const offset = tf.scalar(255.0);
    // tensor = tf.scalar(1.0).sub(tensor.div(offset));
    tensor.print();
    //We add a dimension to get a batch shape 
    // const batched = normalized.expandDims(0)

 

    tensor = tf.expandDims(tensor, axis=0);
    console.log(tensor);
    console.log(tensor.shape);

    parsifiedInfo = await model.predict(tensor).dataSync();


    console.log(parsifiedInfo)

    normalizedPrediction = ((
      parsifiedInfo[0]*1 +
       parsifiedInfo[1]*10 +
        parsifiedInfo[2]*2 +
         parsifiedInfo[3]*3 +
          parsifiedInfo[4]*4 +
           parsifiedInfo[5]*5 +
            parsifiedInfo[6]*6 +
             parsifiedInfo[7]*7 +
              parsifiedInfo[8]*8 +
               parsifiedInfo[9]*9))/10;
    let percentage = normalizedPrediction*100;

    console.log(percentage);
    console.log(normalizedPrediction);

  }

  const imageDiv = document.createElement('div');
  imageDiv.appendChild(img);
    document.body.appendChild(imageDiv);

  model.summary();



//   console.log("img", img);




// document.getElementById('micro-out-div').innerText =

}

run();
