const input = document.getElementById("upload");
const canvasOriginal = document.getElementById("canvasOriginal");
const canvasModified = document.getElementById("canvasModified");
const ctxOriginal = canvasOriginal.getContext("2d");
const ctxModified = canvasModified.getContext("2d");

input.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if(!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = () => {
        //Ajustamos el tamaño del canvas
        canvasOriginal.width = canvasModified.width = img.width;
        canvasOriginal.height = canvasModified.height = img.height;
        ctxOriginal.drawImage(img, 0, 0, img.width, img.height);

        //Obtenemos los datos de la img
        const imageData = ctxOriginal.getImageData(0, 0, img.width, img.height);
        const {width, height, data} = imageData;

        //Lo convertimos en tensor y modificamos el canal verde del RGB
        const modifiedTensor = tf.tidy( () => {
            let tensor = tf.tensor1d(data, 'int32').reshape([height, width, 4]); //RGBA
            //El -1 en la funcion slice(start, size) significa que tome todo el tamaño disponible de esa dimensión
            let rgb = tensor.slice([0, 0, 0], [-1, -1, 3]); //Obtenemos solo el RGB sin el alpha

            //Aumentamos un 25% del valor del canal G
            let green = rgb.slice([0, 0, 1], [-1, -1, 1]); //Para extraer el canal G
            //Se encarga de que al incrementar el 25% el valor G y que el valor final esta entre 0 y 255
            // clipByValue(0, 255) es lo que nos permite limitar los valores
            green = green.mul(1.25).clipByValue(0, 255);

            //Reconstruimos la img con el nuevo valor del canal G 
            //el 2 en la funcion representa la dimension en la q se está concatenando
            rgb = tf.concat([
                rgb.slice([0, 0, 0], [-1, -1, 1]),
                green,
                rgb.slice([0, 0, 2], [-1, -1, 1])
            ], 2);

            //Generamos un canal alpha de 255 para añadirlo
            let alpha = tf.ones([height, width, 1]).mul(255);
            return tf.concat([rgb, alpha], 2).cast('int32');
        });

        //convertir el tensor modificado a imageData
        modifiedTensor.data()
            .then( modifiedData => {
                //Uint8ClampedArray realiza una copia dato a dato del vector
                const newImageData = new ImageData( new Uint8ClampedArray(modifiedData), width, height);
                ctxModified.putImageData(newImageData, 0, 0);
            });
        
        //Liberamos la memoria
        modifiedTensor.dispose();
    }
});
