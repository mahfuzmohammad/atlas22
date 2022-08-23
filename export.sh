#!/usr/bin/env bash

./build.sh

docker save seg | gzip -c > seg.tar.gz
