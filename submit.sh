echo "Start"

tag=aicregistry:5000/${USER}:surgrefexp
echo "ghjdfi"

docker build .          \
  --tag ${tag}                  \
  --build-arg USER_ID=$(id -u)  \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}      \
  --network=host                
  # --progress=plain --no-cache

docker push ${tag}

runai submit            \
    --backoff-limit 0   \
    --name ${1}         \
    --image ${tag}      \
    --gpu 1            \
    --project ${USER}   \
    --volume /nfs:/nfs  \
    --large-shm         \
    --run-as-user       \
    --command -- ${@:2} \


