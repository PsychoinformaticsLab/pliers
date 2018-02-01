#! /bin/bash

if [[ -v GOOGLE_APPLICATION_CREDENTIALS ]]; then
  openssl aes-256-cbc -K $encrypted_a0a62c26415d_key -iv $encrypted_a0a62c26415d_iv
  -in pliers/tests/credentials/google.json.enc -out pliers/tests/credentials/google.json -d;
 fi