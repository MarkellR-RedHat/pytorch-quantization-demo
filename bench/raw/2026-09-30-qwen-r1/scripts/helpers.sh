NS=<namespace>

inpod() { oc exec -n $NS "$1" -c kserve-container -- bash -lc "$2"; }
podof() { oc get pods -n $NS -l serving.kserve.io/inferenceservice=$1 -o jsonpath='{.items[0].metadata.name}'; }
waitready() { oc wait -n $NS --for=condition=Ready isvc/$1 --timeout=30m; }
warmup() { for i in 1 2 3 4 5; do inpod $1 "curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":32}'" > /dev/null; done; }
getsharegpt() { inpod "$1" "curl -sL -o /tmp/sharegpt.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json && ls -la /tmp/sharegpt.json"; }
specmetrics() { inpod "$1" "curl -s localhost:8080/metrics | grep -E '^vllm:spec_decode'" > results-qwen/logs/spec-metrics-$2.txt; }
gpuutil() { oc exec -n $NS $1 -c kserve-container -- nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 2 > results-qwen/logs/$2-gpu-util-c64.csv; }
