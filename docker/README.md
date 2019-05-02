# pliers-docker
Docker setup for pliers

One tricky aspect of running pliers is that it requires credentials for several different web APIs,
which we certainly don't want to include in the container!  We deal with this by requiring the user
to create a bash script to set the appropriate environment variables for those API credentials,
and then source it upon executing the container.  There are almost certainly more elegant ways to do this
but it works well enough.

To use it:

1. Build the docker container. Run the following in the pliers root directory.

    ```
    docker build --tag pliers --file docker/Dockerfile .
    ```

2. Create a file that contains all of the relevant API keys, called env.list:

```
WIT_AI_API_KEY=abc123
IBM_USERNAME=joe@schmo.edu
IBM_PASSWORD=xyzabc
INDICO_APP_KEY=abcyyz
GOOGLE_APPLICATION_CREDENTIALS=/root/share/googleapi.json
CLARIFAI_API_KEY=hhhvvv
```
This should be obvious, but just to be sure: NEVER check this file into a github repository, unless you want to pay for free cloud computing for a cybercriminal.

3. Obtain a service account key for your Google Cloud account, and save it to a file called googleapi.json

4. Open an interactive session on the docker container:

    docker run --env-file ./env.list -it -v /path/to/pliers:/root/share pliers

where /path/to/pliers contains the files generated above.

At this point you should have a python (3.5) environment that is ready for you to start using pliers.
