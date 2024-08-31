---
layout: post
title:  "The Guide to Docker ARG, ENV and .env"
date:   2020-12-27 00:00:00 +0800
# categories: main
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

- ENV are available during build-time and run-time
- Use this to pass in secrets during run-time and avoid hard-coding them in build-time
- ENV cannot be declared before the FROM syntax
- In a multi-stage build, ENV persists across all stages
- Takes precedence over ARG of the same variable name. For example, in a Dockerfile where the same variable name is defined by both ENV and ARG, the value for ENV will be used instead of ARG


`ARG`

- ARG are also known as build-time environment variables as they are only available during build-time, and not during run-time
- Do not use this to set secrets because build-time values are visible to any user of the image using the docker history command
- ARG can be declared before the FROM syntax
- In a multi-stage build, ARG does not persist beyond the first stage
- During build-time, you can override the ARG variables with the flag --build-arg <varname>=<value> to build image with different variables. Note: this does not work if there exists ENV configured with the same variable name, see section below on precedence


### ENV takes precedence over ARG
In the following example, the same variable SOME_VAR is defined by both ARG and ENV. As the value from ENV takes precedence over ARG, building the image using the command docker build --no-cache -t demo . would print Build-time: SOME_VAR is set to env-value in one of the layers as it prints value from the ENV instead. This means that value from ARG is ignored.