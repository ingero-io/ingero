package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// completionCmd generates shell completion scripts for bash, zsh, fish, and powershell.
var completionCmd = &cobra.Command{
	Use:   "completion [bash|zsh|fish]",
	Short: "Generate completion script",
	Long: fmt.Sprintf(`To load completions:

Bash:

  $ source <(%[1]s completion bash)

  # To load completions for every new session, run once:
  $ %[1]s completion bash > /etc/bash_completion.d/%[1]s

Zsh:

  # If shell completion is not already enabled in your environment,
  # you will need to enable it.  You can execute the following once:

  $ echo "autoload -U compinit; compinit" >> ~/.zshrc

  # For the current user, run once:
  $ mkdir -p ~/.zsh/completions
  $ %[1]s completion zsh > ~/.zsh/completions/_%[1]s
  $ echo 'fpath=(~/.zsh/completions $fpath)' >> ~/.zshrc  

  # For all users, run as root once:
  # %[1]s completion zsh > /usr/local/share/zsh/site-functions/_%[1]s

Fish:

  $ %[1]s completion fish | source

  # For the current user, run once:
  $ mkdir -p ~/.config/fish/completions
  $ %[1]s completion fish > ~/.config/fish/completions/%[1]s.fish

  # For all users, run as root once:
  # %[1]s completion fish > /etc/fish/completions/%[1]s.fish

`, "ingero"),
	DisableFlagsInUseLine: true,
	ValidArgs:             []string{"bash", "zsh", "fish"},
	Args:                  cobra.MatchAll(cobra.MaximumNArgs(1), cobra.OnlyValidArgs),
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			cmd.Help()
			return
		}
		switch args[0] {
		case "bash":
			cmd.Root().GenBashCompletion(os.Stdout)
		case "zsh":
			cmd.Root().GenZshCompletion(os.Stdout)
		case "fish":
			cmd.Root().GenFishCompletion(os.Stdout, true)
		}
	},
}

func init() {
	rootCmd.AddCommand(completionCmd)
}
