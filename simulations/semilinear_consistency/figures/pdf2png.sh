#!/bin/bash

for pdfile in *.pdf ; do
  pdftoppm "${pdfile}" "${pdfile%.*}" -png
done

