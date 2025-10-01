class PCMProcessor extends AudioWorkletProcessor {
  process(inputs, outputs) {
    if (!inputs.length || !inputs[0].length) return true;

    const input = inputs[0][0];              // first channel, Float32Array
    const int16 = new Int16Array(input.length);

    for (let i = 0; i < input.length; i++) {
      let s = Math.max(-1, Math.min(1, input[i])); // clamp
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;  // convert to PCM16
    }

    this.port.postMessage(int16.buffer, [int16.buffer]); // transfer buffer
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
