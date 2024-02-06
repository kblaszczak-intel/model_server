pipeline {
    agent {
      label 'ovmsoncommit'
    }
    stages {
        stage('Configure') {
          steps {
            script {
              checkout scm
              shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
              echo shortCommit
            }
          }
        }

        stage('style check') {
            steps {
                sh 'make style'
            }
        }

        stage('sdl check') {
            steps {
                sh 'make sdl-check'
            }
        }

        stage("Run smoke and regression tests on commit") {
          steps {
              sh 'make docker_build'
          }    
        }
    }
}
