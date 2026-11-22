// Starts the program given as its arguments and waits for it. Under launchd, macOS attributes a child's file access to
// the service's own program, so the Removable Volumes permission is given once to this launcher instead of to every new
// origami build. Never rebuild, re-sign or move the built binary: the permission is tied to that exact file at its path.
// Build once: cc -O2 -o origami-launcher Tools/launcher/origami-launcher.c (docs/integration.md says how to use it).
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: origami-launcher program [arguments]\n");
        return 64;
    }
    int forwarded[] = {SIGTERM, SIGINT, SIGHUP, SIGUSR1, SIGUSR2};
    for (unsigned i = 0; i < sizeof forwarded / sizeof *forwarded; i++) signal(forwarded[i], forward);
    int error = posix_spawn(&child, argv[1], NULL, NULL, argv + 1, environ);
    if (error) {
        fprintf(stderr, "origami-launcher: %s: %s\n", argv[1], strerror(error));
        return 71;
    }
    int status;
    while (waitpid(child, &status, 0) < 0)
        if (errno != EINTR) return 71;
    // Ends as the program ended, so launchd's KeepAlive sees a crash as a crash.
    if (WIFSIGNALED(status)) {
        signal(WTERMSIG(status), SIG_DFL);
        raise(WTERMSIG(status));
        return 128 + WTERMSIG(status);
    }
    return WEXITSTATUS(status);
}
