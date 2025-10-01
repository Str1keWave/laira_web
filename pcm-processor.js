class PCMProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {

    //checks there's real audio to work with
    if (inputs.length === 0){
      return true}
    if (inputs[0].length === 0){
      return true
    }
    
    const input = inputs[0][0];

    //convert 32 -> 16
    let int16Block = new Int16Array(input.length)

    //scale each to the 16 int limit
    for (let i = 0; i < input.length; i++){
      let v = input[i]
      if (v > 1) v = 1
      else if (v < -1) v = -1
      // scale
      int16Block[i] = (v < 0 ? v * 0x8000 : v * 0x7FFF)  // -32768..32767
    }
    
     this.port.postMessage({
        type: "pcm16",
        buffer: int16Block.buffer
      }, [int16Block.buffer])
  
    // as this is a source node which generates its own output,
    // we return true so it won't accidentally get garbage-collected
    // if we don't have any references to it in the main thread
    return true;
  }

}

registerProcessor('pcm-processor', PCMProcessor)
  
