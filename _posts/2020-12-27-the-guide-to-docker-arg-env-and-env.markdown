---
layout: post
title:  "The Guide to Docker ARG, ENV and .env"
date:   2020-12-27 00:00:00 +0800
categories: main
---

Many times, developers have been left scratching their heads figuring out the best way to pass in variables at different stages of development and deployment. I, myself, are guilty of that. To solve this once and for all, I decided to experiment and note down my observations in this article. 🤗

For those in a hurry, here’s a table summarizing the content of this post. You can find a higher resolution [here](https://docs.google.com/presentation/d/1QqCYdF67zEpRg724BlFceDhg_ntPmiZ5rgmSq_vlssk/edit).

![Image]({{ site.baseurl }}/assets/images/2020-12-27/1.png){: width="100%" }

### Setting Environment Variable in Docker
In this section, I present you with __four different ways__ you can provide values to the environment variables to your docker image during build-time using docker-compose.

### 1. Docker Dot-Env File (.env)

The `.env` file is probably the most common way to set the environment variables. By storing the variables in a file, you can track their changes using Git. Typically, the content of `.env` is written in the follow notation:

{% highlight docker %}
VARIABLE_ONE=ONE
VARIABLE_TWO=TWO
{% endhighlight %}

With the `.env` file in the current directory and the following `docker-compose.yml` file, you can run the command `docker-compose up -d` to spin up a container. Behind the scenes, it will read the variables in `.env` file and print the value to console as instructed by `command`. To verify, run `docker logs ubuntu` and you will see the variable `ONE` being logged.

Fun fact: the double `$$` is used if you need a literal dollar sign in a docker-compose file. This also prevents Compose from interpolating a value, so a `$$` allows you to refer to environment variables that you don’t want processed by Compose. This is documented [here](https://docs.docker.com/compose/compose-file/compose-file-v3/).

{% highlight docker %}
version: "3"
services:
  hello_world:
    container_name: ubuntu
    image: ubuntu:latest
    env_file: .env # optional, change this if your filename is different
    command: '/bin/sh -c "echo $$VARIABLE_ONE"'
{% endhighlight %}

### 2. Using host’s environment variable
Alternatively, if you do not wish to create a `.env` file, and instead want to use an existing environment variable in your host, you can also do so with the following `docker-compose.yml` file. This way, Docker will read in your host’s environment variable and pass it to the container. However, I do not recommend using this method as it may make it hard for you to debug.

{% highlight docker %}
    $ export VARIABLE_ONE=ONE
{% endhighlight %}

{% highlight docker %}
version: "3"
services:
  hello_world:
    container_name: ubuntu
    image: ubuntu:latest
    command: '/bin/sh -c "echo $$VARIAB
{% endhighlight %}

### 3. Docker ENV
Another way of setting environment variables is to define it directly in your `docker-compose.yml` file using the `environment:` syntax.

{% highlight docker %}
version: "3"
services:
  hello_world:
    container_name: ubuntu
    image: ubuntu:latest
    environment:
      VARIABLE_ONE: ONE
    command: '/bin/sh -c "echo $$VARIABLE_ONE"'
{% endhighlight %}

### 4. Using Shell Parameter Expansion
The last way is to set the environment variable within the parameter itself. Using the Shell Parameter Expansion feature, `${VARIABLE_ONE:-ONE}` would default to the value `ONE` if it is not overridden during run-time. For more information about this behavior, see [bash reference](https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html) and the documentation by Docker [here](https://docs.docker.com/compose/compose-file/compose-file-v3/).

{% highlight docker %}
version: "3"
services:
  hello_world:
    container_name: ubuntu
    image: ubuntu:latest
    command: '/bin/sh -c "echo ${VARIABLE_ONE:-ONE}"'
{% endhighlight %}


## Two types of variables in Docker — ARG and ENV
There are two types of environment variables in Docker. In a Dockerfile, they come in the syntax of `ARG` and `ENV`. In a Docker Compose file, they are `args:` and `environment:`. I have summarized the differences below in point-form for easy reference.

`ENV`

- `ENV` are __available during build-time and run-time__
- Use this to pass in secrets during run-time and avoid hard-coding them in build-time
- `ENV` cannot be declared before the `FROM` syntax
- In a multi-stage build, `ENV` persists across all stages
- Takes precedence over `ARG` of the same variable name. For example, in a Dockerfile where the same variable name is defined by both `ENV` and `ARG`, the value for `ENV` will be used instead of `ARG`


`ARG`

- `ARG` are also known as build-time environment variables as they are __only available during build-time, and not during run-time__
- Do not use this to set secrets because build-time values are visible to any user of the image using the `docker history` command
- `ARG` can be declared before the `FROM` syntax
- In a multi-stage build, `ARG` does not persist beyond the first stage
- During build-time, you can override the `ARG` variables with the flag `--build-arg <varname>=<value>` to build image with different variables. Note: this does not work if there exists `ENV` configured with the same variable name, see section below on precedence

### ENV takes precedence over ARG
In the following example, the same variable `SOME_VAR` is defined by both `ARG` and `ENV`. As the value from `ENV` takes precedence over `ARG`, building the image using the command docker `build --no-cache -t demo`. would print `Build-time: SOME_VAR` is set to `env-value` in one of the layers as it prints value from the `ENV` instead. This means that value from `ARG` is ignored.

{% highlight docker %}
FROM ubuntu
ENV SOME_VAR=env-value
ARG SOME_VAR=arg-value
RUN echo "Build-time: SOME_VAR is set to ${SOME_VAR:-}"
CMD ["bash", "-c", "echo Run-time: SOME_VAR is set to ${SOME_VAR:-}"]
{% endhighlight %}


{% highlight bash %}
$ docker build --no-cache -t demo .

### REMOVED ###
Step 2/5 : ENV SOME_VAR=env-value
---> Running in ed8e108898e4
Removing intermediate container ed8e108898e4
---> 43da9e8e6dd6
Step 3/5 : ARG SOME_VAR=arg-value
---> Running in fb03d6097fb1
Removing intermediate container fb03d6097fb1
---> 853bb58e415e
Step 4/5 : RUN echo "Build-time: SOME_VAR is set to ${SOME_VAR:-}"
---> Running in 709fd76468fa
Build-time: SOME_VAR is set to env-value
### REMOVED ###

$ docker run --rm demo
Run-time: SOME_VAR is set to env-value
{% endhighlight %}

Also, building the image with the flag `--build-arg SOME_VAR=new-value` will have no effect as well.

{% highlight bash %}
$ docker build --build-arg SOME_VAR=new-value --no-cache -t demo .

### REMOVED ###
Step 2/5 : ENV SOME_VAR=env-value
 ---> Running in 10d7cb325994
Removing intermediate container 10d7cb325994
 ---> dcab47c67952
Step 3/5 : ARG SOME_VAR=arg-value
 ---> Running in 265c48974673
Removing intermediate container 265c48974673
 ---> 3fab670db0bf
Step 4/5 : RUN echo "Build-time: SOME_VAR is set to ${SOME_VAR:-}"
 ---> Running in 0339a0690972
Build-time: SOME_VAR is set to env-value
Removing intermediate container 0339a0690972
 ---> aa61afbb01cd
### REMOVED ###

$ docker run --rm demo
Run-time: SOME_VAR is set to env-value
{% endhighlight %}

### Multi-Stage Image Build with ONBUILD syntax

The concept of `ONBUILD` allows you to declare `ARG` and `ENV` in a stage and let the values be available only in the subsequent stages.

In the Dockerfile example below, I’ve declared four environment variables in the first stage, namely `VAR_ENV`, `VAR_ARG`, `VAR_ENV_ONBUILD` and `VAR_ARG_ONBUILD`.

{% highlight bash %}
# First stage
FROM debian as base
ENV VAR_ENV=env_value
ARG VAR_ARG=arg_value
ONBUILD ENV VAR_ENV_ONBUILD=onbuild_env_value
ONBUILD ARG VAR_ARG_ONBUILD=onbuild_arg_value
RUN echo "First stage build time:"; env | grep VAR_
# Second stage
FROM base
RUN echo "Second stage build time:"; env | grep VAR_
# Third stage
FROM base
RUN echo "Third stage build time:"; env | grep VAR_
CMD ["bash", "-c", "echo At runtime; env | grep VAR_"]
{% endhighlight %}

During build-time, notice that in Step 6 (first stage), only `VAR_ARG` and `VAR_ENV` are printed. However, in Step 8 (second stage), `VAR_ARG_ONBUILD`, `VAR_ENV_ONBUILD` and `VAR_ENV` are printed except `VAR_ARG`. This proves that `VAR_ARG` does not persist beyond its own first stage and that `VAR_*_ONBUILD` are only available in subsequent second and third stages (see Step 10 for third stage).


{% highlight bash %}
$ docker build --no-cache -t demo-onbuild .

### REMOVED ###
Step 6/11 : RUN echo "First stage build time:"; env | grep VAR_
 ---> Running in 2059bd2e7860
First stage build time:
VAR_ARG=arg_value
VAR_ENV=env_value
Removing intermediate container 2059bd2e7860
 ---> ec2a853d2c02
### REMOVED ###
Step 8/11 : RUN echo "Second stage build time:"; env | grep VAR_
 ---> Running in c6002665e972
Second stage build time:
VAR_ARG_ONBUILD=onbuild_arg_value
VAR_ENV_ONBUILD=onbuild_env_value
VAR_ENV=env_value
Removing intermediate container c6002665e972
 ---> b8e8b99d65cd
### REMOVED ###
Step 10/11 : RUN echo "Third stage build time:"; env | grep VAR_
 ---> Running in c71c1773ee98
Third stage build time:
VAR_ARG_ONBUILD=onbuild_arg_value
VAR_ENV_ONBUILD=onbuild_env_value
VAR_ENV=env_value
Removing intermediate container c71c1773ee98
 ---> 005c9e64aeea
Step 11/11 : CMD ["bash", "-c", "echo At runtime; env | grep VAR_"]
 ---> Running in fbdd2385224a
Removing intermediate container fbdd2385224a
 ---> c96e28248334
Successfully built c96e28248334
Successfully tagged demo-onbuild:latest
{% endhighlight %}


### Optional Read: Background on Environment Variable vs Shell Variable

In case you are wondering, a __Shell Variable__ is local to a particular instance of the shell, while the __Environment Variables__ are inherited by any program, including from another shell session. This also means that a Shell Variable is a subset of Environment Variables, and is “temporarily” available to the shell session in a sense.

In general, the variables are stored in a key-value pair structure. Shell Variable is set using the command `SOME_SHELL_VAR=shell-var` and Environment Variable is set using export `SOME_ENV_VAR=env-var`, with the extra `export` keyword. There are a few ways to list all the currently defined environment variables and that is by running the command `set`, `printenv` or `env`. However, the shell variables (non-exported) can only be found using the `set` command.

__View all Shell and Environment Variables using `set`__

{% highlight bash %}
$ set
'!'=0
'#'=0
#### REDACTED ####
userdirs
usergroups
watch=(  )
widgets
{% endhighlight %}

__Setting and viewing a Shell Variable__

{% highlight bash %}
$ SAMPLE_SHELL_VAR=shell-var
$ echo $SAMPLE_SHELL_VAR
shell-var
$ set | grep 'SAMPLE_SHELL_VAR' 
SAMPLE_SHELL_VAR=shell-var
{% endhighlight %}

__View all Environment Variables using `printenv`__

{% highlight bash %}
$ printenv
GJS_DEBUG_TOPICS=JS ERROR;JS LOG
SSH_AUTH_SOCK=/run/user/1000/keyring/ssh
### REDACTED ###
XDG_SESSION_TYPE=x11
GNOME_SHELL_SESSION_MODE=ubuntu
HOMEBREW_REPOSITORY=/home/linuxbrew/.linuxbrew/Homebrew
{% endhighlight %}


__Setting and viewing an Environment Variable__

{% highlight bash %}
$ export SAMPLE_ENV_VAR=env-var
$ echo $SAMPLE_ENV_VAR
env-var
$ printenv SAMPLE_ENV_VAR
env-var
{% endhighlight %}

For more detailed information, see the guide from Ubuntu [here](https://help.ubuntu.com/community/EnvironmentVariables).

### Conclusion
By now you should have a good understanding of how environment variables work in Docker — in both forms of ENV and ARG. I hope this serves as a good reference to in your journey to learning and using Docker. 🤗
