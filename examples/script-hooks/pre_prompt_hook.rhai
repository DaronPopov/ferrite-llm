fn pre_prompt(prompt) {
    let system = "[script-hook] Keep answers concise and technical.\n";
    system + prompt
}

fn post_response(response) {
    response + "\n[post-response hook applied]"
}

fn post_chunk(chunk) {
    chunk
}

fn post_logits(candidates) {
    let out = candidates;
    if out.len > 0 {
        out[0]["logit"] = out[0]["logit"] + 5.0;
    }
    out
}

fn gpu_logits_program() {
    "
        x = input([1, 1, 1, 16])
        y = relu(x)
        return y
    "
}
